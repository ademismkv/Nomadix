import pandas as pd
import numpy as np
import cv2
import base64
import gc
import torch
from app.core.model import get_model
from app.core.config import settings
from app.services.grok import get_grok_response

# Load CSVs once at startup
def load_csvs():
    singular_df = pd.read_csv(settings.SINGULAR_CSV)
    combined_df = pd.read_csv(settings.COMBINED_CSV)
    return singular_df, combined_df

singular_df, combined_df = load_csvs()

def analyze_ornament_image(image_bytes: bytes):
    """
    Process image, run YOLO, extract ornaments, and get cultural analysis.
    """
    model = get_model()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    try:
        results = model(img)
        ornament_counts = {}
        unique_ornaments = {}
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                if confidence > 0.5:
                    ornament_counts[class_name] = ornament_counts.get(class_name, 0) + 1
                    if class_name not in unique_ornaments or confidence > unique_ornaments[class_name]['confidence']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped = img[y1:y2, x1:x2]
                        _, buffer = cv2.imencode('.jpg', cropped)
                        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                        unique_ornaments[class_name] = {
                            'class': class_name,
                            'image': cropped_base64,
                            'confidence': confidence
                        }
        detected_ornaments = list(ornament_counts.keys())
        cropped_ornaments = list(unique_ornaments.values())
        prompt = generate_prompt(detected_ornaments)
        response = get_grok_response(prompt)
        return {
            'detected_ornaments': detected_ornaments,
            'ornament_counts': ornament_counts,
            'response': response,
            'cropped_ornaments': cropped_ornaments
        }
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def generate_prompt(ornaments):
    ornaments_lower = [ornament.lower() for ornament in ornaments]
    matching_singular = singular_df[singular_df['name'].str.lower().isin(ornaments_lower)]
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
