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
import gc
import torch

# Try to load environment variables from .env file, but don't fail if it doesn't exist
load_dotenv(override=True)

# Initialize FastAPI app
app = FastAPI(title="Ornament Analysis Service")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global model instance
model = None

def load_model():
    global model
    if model is None:
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load the model from Hugging Face
            model = YOLO('https://huggingface.co/crowded-mind/Nomadix/resolve/main/best.pt')
            
            # Force garbage collection
            gc.collect()
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
    return model

# Load CSV files into DataFrames
try:
    singular_df = pd.read_csv('singular.csv')
    combined_df = pd.read_csv('all_combined_ornaments.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    raise

# Grok API configuration
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    print("Warning: GROK_API_KEY not found in environment variables")

GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def process_image(image_bytes: bytes):
    # Get the global model instance
    global_model = load_model()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    try:
        # Run YOLOv8 detection
        results = global_model(img)
        
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
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clear memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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
    if not GROK_API_KEY:
        print("Error: GROK_API_KEY is not set")
        raise HTTPException(status_code=500, detail="GROK_API_KEY not configured")
        
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
        print(f"Making request to Grok API with prompt: {prompt[:100]}...")
        response = requests.post(GROK_API_URL, headers=headers, json=data)
        print(f"Grok API response status: {response.status_code}")
        print(f"Grok API response content: {response.text[:200]}...")
        
        if response.status_code != 200:
            print(f"Error response from Grok API: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Grok API error: {response.text}")
            
        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            print(f"Unexpected response format: {response_data}")
            raise HTTPException(status_code=500, detail="Unexpected response format from Grok API")
            
        return response_data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Network error calling Grok API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error calling Grok API: {str(e)}")
    except Exception as e:
        print(f"Unexpected error calling Grok API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling Grok API: {str(e)}")

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        print("Image file read successfully")
        
        # Process image with YOLOv8
        detected_ornaments, ornament_counts = process_image(contents)
        print(f"Detected ornaments: {detected_ornaments}")
        print(f"Ornament counts: {ornament_counts}")
        
        if not detected_ornaments:
            return {"response": "No ornaments detected in the image."}
        
        # Generate prompt and get Grok response
        prompt = generate_prompt(detected_ornaments)
        print(f"Generated prompt: {prompt[:100]}...")
        
        response = get_grok_response(prompt)
        print(f"Got response from Grok API: {response[:100]}...")
        
        return {
            "detected_ornaments": detected_ornaments,
            "ornament_counts": ornament_counts,
            "response": response
        }
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Pre-load the model when starting the server
    load_model()
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 