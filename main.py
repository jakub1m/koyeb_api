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
    PROMPT_SENTIMENT = Here's an improved version of the prompt designed to encourage more consistent results:
"""
As an advanced sentiment analysis system for song lyrics, evaluate the given lyrics (in Polish or English) and return a JSON result. Adhere strictly to these guidelines:

Classification Rules:

Positive (0): Uplifting, joyful, inspiring, or neutral content without offensive elements. Mild melancholy, introspection, and common swear words are acceptable.
Negative (2):

Any offensive or inappropriate content (beyond common swear words), even if used humorously or with a positive message
Slang terms considered disrespectful or offensive (beyond common swear words)
Crude references to body parts
Content promoting harmful behaviors or attitudes
Religious or patriotic themes
Lyrics not in Polish or English
Songs primarily aimed at young children


Neutral (1): Only if sentiment is truly balanced or unclear after analysis, with no offensive elements beyond common swear words


Analysis Priority:

Presence of offensive/inappropriate content (excluding common swear words)
Target audience (child-focused or not)
Religious or patriotic themes
Overall message and emotional tone
Slang usage (beyond common swear words)
Recurring themes and keywords
Metaphors and deeper meanings
School environment appropriateness


Important Notes:

Common swear words alone do not make a song negative
Any offensive content beyond common swear words results in a negative classification
Child-focused songs are always negative
Religious or patriotic songs are always negative
When in doubt, err on the side of caution for school appropriateness


Output Format:
Return this JSON structure:
{
"sentiment": int, // 0 = positive, 2 = negative, 1 = neutral
"sentiment_score": number, // -1.0 to 1.0 (use 0 for neutral)
"confidence": number, // 0.0 to 1.0
"explanation": string // One concise sentence explaining the classification
}
Consistency Check:
Before finalizing your analysis, review these points:

Is the classification consistent with the provided rules?
Have you considered all priority analysis factors?
Is your explanation clear and directly related to the classification?
Does the sentiment_score align with the sentiment classification?
Is your confidence score justified based on the clarity of the classification?



Maintain strict adherence to these guidelines for all evaluations to ensure consistency across assessments.
"""
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
