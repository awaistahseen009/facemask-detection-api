from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service.core.logic.onnx_inference import facemask_detector
from service.core.schema.output import APIOutput
detect_router = APIRouter()

@detect_router.post("/detect", response_model=APIOutput)
def detect(im: UploadFile):
    # check whether the uploaded file is an image
    if im.filename.split(".")[-1] not in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code = 415, detail = "Not an image")

    image = Image.open(BytesIO(im.file.read()))
    image = np.array(image)
    return facemask_detector(image)