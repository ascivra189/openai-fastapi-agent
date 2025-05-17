from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Ascivra AI Agent is alive and ready!"}

@app.post("/generate")
async def generate(request: Request):
    try:
        data = await request.json()

        topic = data.get("topic", "AI Tools")
        platform = data.get("platform", "YouTube")
        tone = data.get("tone", "engaging")
        voice = data.get("voice_style", "friendly narrator")
        audience = data.get("audience", "general public")

        prompt = f"""
You are an elite AI content creator trained by Ascivra.

🎯 TASK:
Generate full content for the platform: {platform}.
Topic: {topic}
Tone: {tone}
Voice Style: {voice}
Target Audience: {audience}

🧠 FORMAT:
Respond with a valid JSON object:
{{
  "title": "...",
  "description": "...",
  "script": "...",
  "caption": "...",
  "cta": "...",
  "visual_prompt": "..."
}}

🔒 RULES:
- Keep it platform-appropriate
- Write like a real human expert with structure, flair, and strategy
- Use short, punchy sentences and natural flow
- Do NOT write any extra commentary. ONLY return the JSON object.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse the JSON response string safely
        raw_output = response.choices[0].message.content.strip()

        try:
            # Evaluate it as a dictionary
            content = eval(raw_output)
        except:
            content = {"error": "Agent response could not be parsed. Raw output:", "raw": raw_output}

        return content

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})