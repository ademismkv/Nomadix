from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.core.model import get_model
from app.services.ornaments import analyze_ornament_image
from app.schemas.analyze import AnalyzeResponse

router = APIRouter()

@router.post("/analyze-image", response_model=AnalyzeResponse, summary="Analyze an image for Kyrgyz ornaments.")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an image file, detects ornaments, and returns analysis and cultural meaning.
    """
    try:
        contents = await file.read()
        result = analyze_ornament_image(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
