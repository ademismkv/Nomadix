from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import os
from typing import List, Optional
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import torch
import base64
import json
from datetime import datetime
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Ornament Analysis Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load CSV files into DataFrames
try:
    singular_df = pd.read_csv('singular.csv')
    combined_df = pd.read_csv('all_combined_ornaments.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    raise

# Load YOLOv8 model
try:
    # Download model from Hugging Face
    model_path = hf_hub_download(
        repo_id="crowded-mind/Nomadix",
        filename="best.pt",
        local_dir="./models"
    )
    print(f"Model downloaded to: {model_path}")
    
    # Load the model with custom settings
    model = YOLO(model_path)
    model.conf = 0.4  # Set confidence threshold
    model.iou = 0.35  # Set IoU threshold
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    raise

# Groq API configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Create contributions directory if it doesn't exist
os.makedirs('contributions', exist_ok=True)

class ImageAnalysisRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class ContributionRequest(BaseModel):
    image_url: str
    detected_ornaments: List[str]
    user_corrections: List[str]
    additional_notes: Optional[str] = None

def process_image(image_data: bytes) -> List[dict]:
    """Process image and return detected ornaments with their counts."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLOv8 detection
    results = model(img)
    
    # Process results
    detected_ornaments = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            if confidence >= model.conf:
                if class_name in detected_ornaments:
                    detected_ornaments[class_name] += 1
                else:
                    detected_ornaments[class_name] = 1
    
    return [{"ornament": k, "count": v} for k, v in detected_ornaments.items()]

def generate_prompt(detected_ornaments: List[dict]) -> str:
    """Generate prompt for LLaMA based on detected ornaments."""
    prompt = "Analyze the cultural significance and meaning of the following ornaments in 3 sentences or less:\n\n"
    
    for item in detected_ornaments:
        ornament = item["ornament"]
        count = item["count"]
        
        # Check singular meanings
        singular_info = singular_df[singular_df['ornament'].str.lower() == ornament.lower()]
        if not singular_info.empty:
            prompt += f"- {ornament} (appears {count} times): {singular_info.iloc[0]['meaning']}\n"
        
        # Check combined meanings
        combined_info = combined_df[combined_df['ornaments'].str.lower().str.contains(ornament.lower())]
        if not combined_info.empty:
            prompt += f"- Combined with other ornaments: {combined_info.iloc[0]['meaning']}\n"
    
    prompt += "\nProvide a concise analysis of these ornaments' cultural significance and meaning in 3 sentences or less."
    return prompt

def get_groq_response(prompt: str) -> str:
    """Get response from Groq API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama2-70b-4096",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise HTTPException(status_code=500, detail="Failed to get response from Groq API")

@app.post("/analyze")
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze image from URL or base64 string."""
    try:
        # Get image data
        if request.image_url:
            response = requests.get(request.image_url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
            image_data = response.content
        elif request.image_base64:
            try:
                image_data = base64.b64decode(request.image_base64)
            except:
                raise HTTPException(status_code=400, detail="Invalid base64 image data")
        else:
            raise HTTPException(status_code=400, detail="Either image_url or image_base64 must be provided")

        # Process image
        detected_ornaments = process_image(image_data)
        
        # Generate analysis
        prompt = generate_prompt(detected_ornaments)
        analysis = get_groq_response(prompt)
        
        return {
            "detected_ornaments": detected_ornaments,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contribute")
async def contribute_data(contribution: ContributionRequest):
    """Store user contributions for model improvement."""
    try:
        # Create contribution record
        contribution_data = {
            "timestamp": datetime.now().isoformat(),
            "image_url": contribution.image_url,
            "detected_ornaments": contribution.detected_ornaments,
            "user_corrections": contribution.user_corrections,
            "additional_notes": contribution.additional_notes
        }
        
        # Save to file
        filename = f"contributions/contribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(contribution_data, f, indent=2)
        
        return {"status": "success", "message": "Contribution saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

def get_groq_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
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
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    
    # Process image with YOLOv8
    detected_ornaments, ornament_counts = process_image(contents)
    
    if not detected_ornaments:
        return {"response": "No ornaments detected in the image."}
    
    # Generate prompt and get LLaMA response
    prompt = generate_prompt(detected_ornaments)
    response = get_groq_response(prompt)
    
    return {
        "detected_ornaments": detected_ornaments,
        "ornament_counts": ornament_counts,
        "response": response
    }

if __name__ == "__main__":
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 