from pydantic import BaseModel
from typing import List, Dict, Any

class CroppedOrnament(BaseModel):
    class_: str
    image: str  # base64 encoded
    confidence: float

class AnalyzeResponse(BaseModel):
    detected_ornaments: List[str]
    ornament_counts: Dict[str, int]
    response: str
    cropped_ornaments: List[Any]  # List[CroppedOrnament] if you want strict typing
