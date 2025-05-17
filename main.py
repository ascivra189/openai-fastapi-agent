from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv() 
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY")) 

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI OpenAI Script Generator!"}

@app.post("/generate")
async def generate(request: Request):
    try:    
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

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    