from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import logging

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

# Set your Groq API key directly here (Replace with your actual key)
client = Groq(api_key="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")

# Request model for handling input
class PlantAnalysisRequest(BaseModel):
    user_input: str

# Endpoint for analyzing plant-related queries
@app.post("/analyze-plant")
def analyze_plant(data: PlantAnalysisRequest):
    try:
        # Print the user's input for debugging
        logging.info(f"User input: {data.user_input}")

        # Define a system prompt for the medical diagnosis assistant
        prompt = (
            "You are a highly trained medical diagnosis assistant. If the user provides symptoms, "
            "medical history, or a description of a health condition, analyze it using your medical dataset "
            "and expertise, then give a clear, concise, and accurate response. "
            "If the user asks something unrelated to health or medicine, politely inform them: "
            "'I am well trained to answer only medical-related questions. Please provide a health-related query.'"
        )

        # Call Groq's chat API with the user input and system prompt
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": data.user_input},
            ],
            model="llama-3.3-70b-versatile",  # Adjust model name if needed
        )

        # Get the response from Groq API and return it
        response = chat_completion.choices[0].message.content
        logging.info(f"Response: {response}")
        return {"response": response}

    except Exception as e:
        # Catch any error and log it for debugging
        logging.error(f"Error: {e}")
        return {"error": str(e)}

