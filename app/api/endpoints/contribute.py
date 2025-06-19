from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
import os
from datetime import datetime

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.post("/contribute", summary="Contribute a new ornament image and metadata.")
async def contribute_ornament(
    file: UploadFile = File(...),
    label: str = Form(...),
    description: str = Form(None)
):
    """
    Saves the uploaded image to data/uploads and returns the file path.
    """
    try:
        contents = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(' ', '_')
        filename = f"{timestamp}_{safe_label}.jpg"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, 'wb') as f:
            f.write(contents)
        return {"status": "success", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
