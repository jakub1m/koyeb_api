import os
import logging
import json
from typing import Optional, Dict, Any, List
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

api_keys = os.getenv("Api_keys", "").split(",")  # Get API keys from environment variable
api_keys = [key.strip() for key in api_keys if key.strip()]  # Clean and prepare the list
print(api_keys)

class ApiKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.lock = asyncio.Lock()

    async def get_next_key(self) -> str:
        async with self.lock:
            key = self.api_keys[self.current_key_index]
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            return key

class GeminiApi:
    PROMPT_SENTIMENT = """Create a system for evaluating song lyrics for a high school radio station (students aged 16-21). The system should classify songs into three categories: positive sentiment (can be played), neutral sentiment (requires manual review), or negative sentiment (automatically rejected).
Evaluation criteria:

Reject content that is explicitly offensive or promotes harmful behavior
Allow content that discusses personal struggles, emotions, or social issues in a non-harmful way
Be lenient towards mild stereotypes or generational comments if the overall message is positive
Block content that is explicitly patriotic or religious
Block songs typically intended for young children
Consider the overall tone and message of the song
Disregard profanity (it's filtered separately)

Analyze the following song lyrics and classify them according to the above criteria. Return the response in the following JSON format:
{
"overall_sentiment": string, // "positive", "negative", or "neutral"
"sentiment_score": number, // -1 to 1 (0 for neutral)
"confidence": number, // 0 to 1
"explanation": string // One concise sentence justifying the assessment
}
Remember to always provide the response in the specified JSON format, without any additional comments or explanations outside the structure. Songs that express personal struggles, emotions, life reflections, or social commentary in a constructive or artistic way, without being explicitly harmful or promoting negative stereotypes, should be classified as positive. Be cautious with humor that might be perceived as offensive or that reinforces negative stereotypes about appearance or other personal characteristics. Classify as negative any content that mocks or demeans individuals based on their physical appearance or other personal traits, even if presented in a humorous context.
Song lyrics to analyze:"""

    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        self.model = model
        self.api_key_manager = ApiKeyManager(api_keys)
        self._init_model()
    
    def _init_model(self) -> None:
        try:
            self.model_instance = genai.GenerativeModel(self.model, system_instruction=self.PROMPT_SENTIMENT)
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
    
    async def sentiment_analysis(self, lyrics: str, title: str) -> Optional[Dict[str, Any]]:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                api_key = await self.api_key_manager.get_next_key()
                logger.info(f"Using API key: {api_key}")
                genai.configure(api_key=api_key)
                
                response = await self.model_instance.generate_content_async(
                    lyrics,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE 
                    }
                )

                if response is None:
                    logger.warning(f"Response is null for API key {api_key}, indicating quota issues.")
                    continue

                json_content = response.candidates[0].content.parts[0].text
                
                json_content = re.sub(r'```json\s*|\s*```', '', json_content)
                
                match = re.search(r'\{.*?\}', json_content, re.DOTALL)
                if match:
                    json_content = match.group(0)
                
                result = json.loads(json_content)
                if isinstance(result, dict):
                    return result
                else:
                    logger.error(f"Unexpected response format: {json_content}")
                    return None
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with API key {api_key}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait for 1 second before retrying
                else:
                    logger.error("All attempts to communicate with Gemini failed")
                    return None
                
class SentimentRequest(BaseModel):
    lyrics: str
    title: str

gemini_api = GeminiApi()
@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    try:
        result = await gemini_api.sentiment_analysis(request.lyrics, request.title)
        
        if result is None:
            raise HTTPException(status_code=503, detail="Failed to communicate with Gemini API after multiple attempts")
        
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
