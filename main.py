from fastapi import FastAPI
import openai
import os
from dotenv import load_dotenv

load_dotenv() #Load .env variables

app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

    
@app.get("/generate")
async def generate(request:Request):
    data = await request.json()
    topic = data.get("topic", "AI Tools")
    platform = data.get("platform","YouTube")

    prompt = f"Write a complete script for a {platform} video about: {topic}"

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )   
        
    return {"script": response.choices[0].message.content}