from ultralytics import YOLO
import torch
import gc

_model = None

MODEL_URL = 'https://huggingface.co/crowded-mind/Nomadix/resolve/main/best.pt'

def get_model():
    """
    Loads and returns the YOLOv8 model singleton.
    """
    global _model
    if _model is None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _model = YOLO(MODEL_URL)
            gc.collect()
        except Exception as e:
            raise RuntimeError(f"Error loading YOLOv8 model: {e}")
    return _model
