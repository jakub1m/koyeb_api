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
    PROMPT_SENTIMENT ="""As an advanced sentiment analysis system for song lyrics, evaluate the given lyrics (in Polish or English) and return a JSON result. Be extremely strict and uncompromising in your assessments, with zero tolerance for offensive content, regardless of humor or intended message. Rules:

Positive (0):
- Uplifting, joyful, or inspiring content with absolutely no offensive elements
- Songs expressing hope, resilience, or personal growth using entirely respectful language
- Love songs with respectful, non-explicit content
- Mild use of common swear words is acceptable if not directed at individuals

Neutral (1):
- Songs with a perfect balance of positive and negative elements, without any potentially offensive content
- Purely descriptive or narrative content without emotional charge or potentially offensive language

Negative (2):
- ANY use of derogatory language, slurs, or offensive terms (beyond common swear words)
- ANY form of body shaming or mocking physical appearances, NO EXCEPTIONS
- ANY humor targeting individuals or groups based on physical characteristics
- Promotion of harmful behaviors, attitudes, or stereotypes, even if presented humorously
- Use of crude or disrespectful metaphors or comparisons, regardless of context
- Explicit sexual content or graphic descriptions of violence
- Religious or overtly political content
- Children's songs or content primarily aimed at very young audiences
- Lyrics not in Polish or English

Critical Points:
- Humor is NEVER an excuse for offensive content. ANY humorous content that involves body shaming, mocking appearances, or use of derogatory language results in an automatic negative classification
- The intended message, moral, or "lesson" DOES NOT excuse the use of offensive language or harmful stereotypes
- ANY instance of body shaming or mocking physical appearances, even if used for contrast or to highlight a positive change, results in a negative classification
- Be extremely wary of "twist" endings or morals that seem to justify earlier offensive content - the presence of offensive content overrides any positive message

Return this JSON structure:
{
  "sentiment": int, // "positive" = 0, "negative" = 2, "neutral" = 1
  "sentiment_score": number, // -1 to 1 (0 for neutral)
  "confidence": number, // 0 to 1
  "explanation": string // One concise sentence summarizing the rationale, focusing on the presence of any offensive elements
}

Ensure absolute consistency and zero tolerance. ANY presence of body shaming, derogatory language, or mocking of physical appearances results in a negative classification, regardless of humor, overall message, or attempts at a positive moral. When in doubt, always classify as negative."""
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.model = model
        self.api_key = api_key
        self._init_model()

    def _init_model(self) -> None:
        try:
            genai.configure(api_key=self.api_key)
            self.model_instance = genai.GenerativeModel(self.model, system_instruction = self.PROMPT_SENTIMENT)
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise

    async def sentiment_analysis(self, lyrics: str, title: str) -> Optional[Dict[str, Any]]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.model_instance.generate_content_async(
                    lyrics,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
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
