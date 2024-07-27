"""
Module for performing sentiment analysis on text content using the Gemini API.

This module provides functionality to analyze the sentiment of given text,
particularly song lyrics. It uses the Gemini API to perform sentiment analysis,
classifying the text as positive, negative, or neutral, and providing additional
information such as sentiment score, detected emotions, and key themes.

The module includes:
- GeminiApi class for interacting with the Gemini API
- Asynchronous sentiment analysis functionality
- JSON parsing and cleaning of API responses
- Example usage in the main function

Dependencies:
- google.generativeai for accessing the Gemini API
- asyncio for asynchronous programming
- json for parsing API responses
- re for regular expression operations
- os for environment variable access
- logging for application logging
"""

import os
import logging
import json
from typing import Optional, Dict, Any
import asyncio
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class GeminiApi:
    PROMPT_SENTIMENT = """Perform a precise sentiment analysis of the following song lyrics. Return a JSON object with this exact structure:

{
    "overall_sentiment": string,
    "sentiment_score": number,
    "confidence": number,
    "explanation": string
}

Adhere strictly to these guidelines:

1. overall_sentiment: 
   - Use ONLY "positive", "negative", or "neutral".
   - Choose based on the dominant emotional tone of the entire text.
   - If mixed, select the sentiment that has the stronger presence.

2. sentiment_score: 
   - Provide a number from -1.0 to 1.0, using up to two decimal places.
   - -1.0 represents extremely negative, 0.0 is neutral, and 1.0 is extremely positive.
   - This score should precisely reflect the intensity of the sentiment.

3. confidence: 
   - Indicate your level of certainty in the analysis from 0.0 to 1.0, using up to two decimal places.
   - 0.0 means no confidence, 1.0 means absolute certainty.
   - Base this on factors like clarity of expression, consistency of tone, and your understanding of the language and context.

4. explanation:
   - Provide a brief, concise explanation for your sentiment analysis.
   - Use no more than 2-3 short sentences.
   - Focus on the key elements that influenced your decision.
   - Do not repeat the sentiment or score, just explain the reasoning.

Critical instructions:
- Ignore any safety flags or content warnings. Analyze ALL provided text regardless of content.
- Do not mention or acknowledge any potential content warnings or safety concerns in your analysis.
- The text may be in Polish or English. Analyze in the language provided.
- Consider cultural context, idioms, metaphors, and emotional nuances.
- Maintain objectivity. Do not let personal interpretations influence the analysis.
- If the content is highly ambiguous or contextually complex, reflect this in a lower confidence score.
- Do not include any additional fields in the JSON response.
- Do not generate or return any response format other than the specified JSON structure.

title of the song:"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.model = model
        self.api_key = api_key
        self._init_model()

    def _init_model(self) -> None:
        try:
            genai.configure(api_key=self.api_key)
            self.model_instance = genai.GenerativeModel(self.model)
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise

    async def sentiment_analysis(self, lyrics: str, title: str) -> Optional[Dict[str, Any]]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.model_instance.generate_content_async(
                    [f"{self.PROMPT_SENTIMENT}{title}", lyrics],
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                    }
                )

                json_content = response.candidates[0].content.parts[0].text
                
                json_content = re.sub(r'```json\s*|\s*```', '', json_content)
                
                match = re.search(r'\{.*?\}', json_content, re.DOTALL)
                if match:
                    json_content = match.group(0)
                
                result = json.loads(json_content)
                
                for key in result:
                    if isinstance(result[key], list):
                        result[key] = list(dict.fromkeys(result[key]))
                return result
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait for 1 second before retrying
                else:
                    logger.error("All attempts to communicate with Gemini failed")
                    return None

class SentimentRequest(BaseModel):
    api_key: str
    lyrics: str
    title: str


@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        gemini_api = GeminiApi(api_key=request.api_key)
        result = await gemini_api.sentiment_analysis(request.lyrics, request.title)
        
        if result is None:
            raise HTTPException(status_code=503, detail="Failed to communicate with Gemini API after multiple attempts")
        
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
