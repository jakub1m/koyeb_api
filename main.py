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
    PROMPT_SENTIMENT = """As an advanced sentiment analysis system for song lyrics, evaluate the given lyrics (in Polish or English) and return a JSON result. Be consistent in your assessments and decisive, avoiding neutral classifications unless truly warranted.
Rules:

Positive: uplifting, joyful, inspiring, or humorously irreverent content. Mild melancholy or introspection is acceptable if the overall message or tone is ultimately positive or relatable. Can be played on school radio if not explicitly offensive.
Negative: clearly depressing, angry, or potentially upsetting content. Includes overtly patriotic, religious, offensive, or inappropriate content for school. Also applies to lyrics not in Polish or English. Automatically reject.
Neutral: only if the sentiment is genuinely balanced or unclear after careful consideration.

Focus on:

Overall emotional tone, message, and artistic intent
Recurring themes and keywords in context
Metaphors, deeper meanings, and cultural references
Age-appropriateness for school environment (high school level)
Presence of explicitly offensive or inappropriate content (high priority)

Avoid:

Overemphasizing mild melancholy or introspection if the overall message or tone is relatable or positive
Inconsistency in assessments
Basing decisions on single words or phrases out of context
Overlooking artistic intent or cultural context
Being overly sensitive to mild irreverence or humorous content

Return this JSON structure:
{
"overall_sentiment": string, // "positive", "negative", "neutral"
"sentiment_score": number, // -1 to 1 (0 for neutral)
"confidence": number, // 0 to 1
"explanation": string // One concise sentence explaining the rationale
}
Ensure consistency across assessments. If the lyrics are not in Polish or English, classify as negative. Prioritize detecting explicitly offensive or inappropriate content. Balance accuracy with the need for clear, decisive categorization, while considering artistic expression and relatable human experiences.
Lyrics to analyze:
"""

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
