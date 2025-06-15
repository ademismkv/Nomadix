from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
import requests
import os
from typing import List
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Ornament Analysis Service")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load CSV files into DataFrames
try:
    singular_df = pd.read_csv('singular.csv')
    combined_df = pd.read_csv('all_combined_ornaments.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    raise

# Load YOLOv8 model from Hugging Face
try:
    model = YOLO('https://huggingface.co/crowded-mind/Nomadix/resolve/main/best.pt')
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    raise

# Grok API configuration
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def process_image(image_bytes: bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLOv8 detection
    results = model(img)
    
    # Extract detected ornaments and count occurrences
    ornament_counts = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                ornament_counts[class_name] = ornament_counts.get(class_name, 0) + 1
    
    # Convert to list of unique ornaments
    detected_ornaments = list(ornament_counts.keys())
    
    return detected_ornaments, ornament_counts

def generate_prompt(ornaments: List[str]) -> str:
    # Convert ornaments to lowercase for case-insensitive matching
    ornaments_lower = [ornament.lower() for ornament in ornaments]
    
    # Find matching individual ornaments (case-insensitive)
    matching_singular = singular_df[singular_df['name'].str.lower().isin(ornaments_lower)]
    
    # Find matching combined ornaments (case-insensitive)
    matching_combined = combined_df[combined_df['Combination Name'].str.lower().isin(ornaments_lower)]
    
    prompt = "You are an expert ornamentologist. Analyze the following ornaments and provide a concise, straightforward explanation in no more than 3 sentences:\n\n"
    
    if not matching_singular.empty:
        prompt += "Individual Ornaments:\n"
        for _, row in matching_singular.iterrows():
            prompt += f"- {row['name']}: {row['en']} (English), {row['ru']} (Russian), {row['kg']} (Kyrgyz)\n"
    
    if not matching_combined.empty:
        prompt += "\nCombined Ornaments:\n"
        for _, row in matching_combined.iterrows():
            prompt += f"- {row['Combination Name']}: {row['Combined Meaning (en)']} (English), {row['Combined Meaning (ru)']} (Russian), {row['Combined Meaning (kg)']} (Kyrgyz)\n"
    
    if matching_singular.empty and matching_combined.empty:
        return "These ornaments are not yet in the knowledge base."
    
    prompt += "\nPlease provide a concise, straightforward explanation of the cultural significance and meaning of these ornaments, combining both individual and combined meanings where applicable. Limit your response to a maximum of 3 sentences."
    return prompt

def get_grok_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert ornamentologist with deep knowledge of Central Asian cultural symbols and their meanings."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Grok API: {str(e)}")

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    
    # Process image with YOLOv8
    detected_ornaments, ornament_counts = process_image(contents)
    
    if not detected_ornaments:
        return {"response": "No ornaments detected in the image."}
    
    # Generate prompt and get Grok response
    prompt = generate_prompt(detected_ornaments)
    response = get_grok_response(prompt)
    
    return {
        "detected_ornaments": detected_ornaments,
        "ornament_counts": ornament_counts,
        "response": response
    }

if __name__ == "__main__":
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 