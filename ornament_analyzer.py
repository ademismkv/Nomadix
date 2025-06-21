from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
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
import base64
import datetime
import json
import boto3
from botocore.exceptions import ClientError
from supabase import create_client, Client

# Try to load environment variables from .env file, but don't fail if it doesn't exist
load_dotenv(override=True)

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

# Global model instance
model = None

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

S3_BUCKET = os.getenv('AWS_S3_BUCKET')

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

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
        unique_ornaments = {}  # Dictionary to store unique ornaments by class
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    ornament_counts[class_name] = ornament_counts.get(class_name, 0) + 1
                    
                    # Only store the highest confidence detection for each class
                    if class_name not in unique_ornaments or confidence > unique_ornaments[class_name]['confidence']:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Crop the ornament
                        cropped = img[y1:y2, x1:x2]
                        
                        # Convert to base64
                        _, buffer = cv2.imencode('.jpg', cropped)
                        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        unique_ornaments[class_name] = {
                            'class': class_name,
                            'image': cropped_base64,
                            'confidence': confidence
                        }
        
        # Convert to list of unique ornaments
        detected_ornaments = list(ornament_counts.keys())
        cropped_ornaments = list(unique_ornaments.values())
        
        return detected_ornaments, ornament_counts, cropped_ornaments
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

# Helper function to upload to Supabase
async def save_contribution_to_supabase(contents: bytes, label: str, description: str = None):
    if not all([os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')]):
        raise HTTPException(
            status_code=500,
            detail="Supabase credentials not configured"
        )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"contributions/{timestamp}_{label.replace(' ', '_')}.jpg"
    image_base64 = base64.b64encode(contents).decode('utf-8')
    try:
        storage_response = supabase.storage.from_('ornaments').upload(
            filename,
            contents,
            {"content-type": "image/jpeg"}
        )
        public_url = supabase.storage.from_('ornaments').get_public_url(filename)
        metadata = {
            'filename': filename,
            'label': label,
            'description': description,
            'timestamp': timestamp,
            'url': public_url
        }
        db_response = supabase.table('contributions').insert(metadata).execute()
        return metadata
    except Exception as e:
        print(f"Error uploading to Supabase: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error uploading contribution"
        )

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        print("Image file read successfully")
        # Process image with YOLOv8
        detected_ornaments, ornament_counts, cropped_ornaments = process_image(contents)
        print(f"Detected ornaments: {detected_ornaments}")
        print(f"Ornament counts: {ornament_counts}")
        if not detected_ornaments:
            return {"response": "No ornaments detected in the image."}
        # Generate prompt and get Grok response
        prompt = generate_prompt(detected_ornaments)
        print(f"Generated prompt: {prompt[:100]}...")
        response = get_grok_response(prompt)
        print(f"Got response from Grok API: {response[:100]}...")
        # Save to Supabase as a contribution
        label = ', '.join([str(o) for o in detected_ornaments]) if detected_ornaments else 'Unknown'
        description = str(response) if response is not None else ''
        print(f"Uploading to Supabase with label: {label}")
        print(f"Uploading to Supabase with description: {description[:100]}")
        metadata = await save_contribution_to_supabase(contents, label, description)
        return {
            "detected_ornaments": detected_ornaments,
            "ornament_counts": ornament_counts,
            "response": response,
            "cropped_ornaments": cropped_ornaments,
            "contribution": metadata
        }
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contribute")
async def contribute_ornament(
    file: UploadFile = File(...),
    label: str = Form(...),
    description: str = Form(None)
):
    try:
        contents = await file.read()
        metadata = await save_contribution_to_supabase(contents, label, description)
        return {
            "status": "success",
            "message": "Thank you for your contribution!",
            "contribution": metadata
        }
    except Exception as e:
        print(f"Error in contribute_ornament: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Pre-load the model when starting the server
    load_model()
    print("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 