from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai  # <-- updated
from openai.types.chat import ChatCompletionMessage
import os
from dotenv import load_dotenv
import json
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# ✅ Set the key for new SDK style
openai.api_key = api_key

# Initialize FastAPI app
app = FastAPI(
    title="AI Content Generator API",
    description="API for generating content for various platforms using OpenAI",
    version="1.0.0",
)

# Add CORS middleware to allow requests from n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PromptInput(BaseModel):
    prompt: str
    use_case: str

SYSTEM_PROMPTS = {
    "youtube": """
You are a world-class YouTube scriptwriter for an AI-driven channel.

🎯 Goal: Write a viral, cinematic YouTube script (5 min ≈ 750–800 words).
🧠 Format the output in JSON as:
{
  "title": "...",
  "scenes": [
    { "text": "...", "voice_text": "...", "image_prompt": "..." }
  ],
  "cta": "..."
}
Respond only with valid JSON.
""",
    "tiktok": """
You are a viral TikTok video scriptwriter.

🎯 Goal: Create a short video script under 60 seconds. Use 3–5 scenes.
🧠 Format the output in JSON as:
{
  "title": "...",
  "scenes": [
    { "text": "...", "voice_text": "...", "image_prompt": "..." }
  ],
  "cta": "..."
}
Respond only with valid JSON.
""",
    "instagram": """
You are a carousel/story video designer for Instagram.

🎯 Goal: Turn content into 5–7 punchy slides for a carousel or story.
🧠 Format the output in JSON as:
{
  "title": "...",
  "scenes": [
    { "text": "...", "voice_text": "...", "image_prompt": "..." }
  ],
  "cta": "..."
}
Respond only with valid JSON.
""",
    "repurpose": """
You are a repurposing strategist turning video scripts into multi-platform content.

🎯 Output JSON for: twitter, instagram, tiktok, reddit, linkedin, youtube_post
🧠 Format:
{
  "twitter": "...",
  "instagram": { "post": "...", "story_prompt": "..." },
  "tiktok": { "script": "...", "visual_prompt": "..." },
  "reddit": "...",
  "linkedin": "...",
  "youtube_post": "..."
}
Respond only with valid JSON.
""",
    "twitter": "You are a viral tweet writer. Return one tweet under 280 characters.",
    "reddit": "You write Reddit posts. Return JSON: { \"title\": \"...\", \"body\": \"...\" }",
    "linkedin": "You write professional posts for LinkedIn. Return plain text.",
    "youtube_post": "You write short, catchy YouTube community post captions. Return plain text."
}

@app.get("/")
def read_root():
    return {"message": "FastAPI AI Agent: Ready for YouTube, TikTok, Instagram, Twitter, Reddit, and more."}

@app.post("/generate")
async def generate_script(data: PromptInput, request: Request):
    try:
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Request from {client_host} for use_case: {data.use_case}")

        use_case = data.use_case.lower().strip()
        if use_case not in SYSTEM_PROMPTS:
            logger.warning(f"Unsupported use_case requested: {use_case}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported use_case: {use_case}. Supported use cases: {', '.join(SYSTEM_PROMPTS.keys())}"
            )

        system_prompt = SYSTEM_PROMPTS[use_case]
        truncated_prompt = data.prompt[:50] + "..." if len(data.prompt) > 50 else data.prompt
        logger.info(f"Processing prompt: {truncated_prompt}")

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": data.prompt}
                ],
                timeout=60
            )

            ai_result = response.choices[0].message.content
            logger.info("Successfully received response from OpenAI")

            try:
                parsed = json.loads(ai_result)
                logger.info("Successfully parsed JSON response")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {str(e)}. Returning raw text.")
                parsed = {"text": ai_result.strip()}

            return {
                "platform": use_case,
                "response": parsed
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
