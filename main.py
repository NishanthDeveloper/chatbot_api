
import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Request model
class PlantAnalysisRequest(BaseModel):
    user_input: str

@app.post("/analyze-plant")
def analyze_plant(data: PlantAnalysisRequest):
    prompt = (
       "You are a highly trained medical diagnosis assistant. If the user provides symptoms, medical history, or a description of a health condition, analyze it using your medical dataset and expertise, then give a clear, concise, and accurate response. If the user asks something unrelated to health or medicine, politely inform them: 'I am well trained to answer only medical-related questions. Please provide a health-related query.'"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": data.user_input},
        ],
        model="llama-3.3-70b-versatile",
    )

    return {"response": chat_completion.choices[0].message.content}