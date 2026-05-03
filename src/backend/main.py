import sys
import os
import json
import base64
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

sys.path.append("/app/model")
sys.path.append("/app/data_management")

from data_augmenter import DataAugmenter


# ── Config ────────────────────────────────────────────────────────────────────

TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://torchserve:8080")
MODEL_NAME     = "blood_cell_detection"


# ── Init ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Blood Cell Detection API")

augmenter = DataAugmenter(
        input_root="/app/data/dataset",  
        output_root="/app/data/aug_data",
        class_map={"RBC": 0, "WBC": 1, "Platelets": 2},
        seed=42,
    )

# ── Models ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request model for symbol detection.

    :param image: Base64-encoded input image, with or without data URI prefix.
    :type image: str
    """
    image: str


class AugmentRequest(BaseModel):
    """Request model for image augmentation.

    :param image: Base64-encoded input image.
    :type image: str
    :param n_variants: Number of augmented variants to generate. Defaults to 5.
    :type n_variants: int
    """
    image: str
    n_variants: int = 5


class BBoxModel(BaseModel):
    """Bounding box coordinates.

    :param x1: Left edge.
    :param y1: Top edge.
    :param x2: Right edge.
    :param y2: Bottom edge.
    """
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionModel(BaseModel):
    """Single detection result.

    :param class_id: Numeric class index.
    :type class_id: int
    :param class_name: Human-readable class label.
    :type class_name: str
    :param confidence: Detection confidence score in ``[0, 1]``.
    :type confidence: float
    :param bbox: Bounding box coordinates.
    :type bbox: BBoxModel
    """
    class_id:   int
    class_name: str
    confidence: float
    bbox:       BBoxModel


class PredictResponse(BaseModel):
    """Response model for symbol detection.

    :param detections: List of detected symbols.
    :type detections: List[DetectionModel]
    """
    detections: List[DetectionModel]


class AugmentResponse(BaseModel):
    """Response model for image augmentation.

    :param variants: List of Base64-encoded augmented images.
    :type variants: List[str]
    """
    variants: List[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Run Blood Cell Detection detection on a Base64-encoded image.

    Forwards the image to TorchServe and returns a list of detected
    symbols with their bounding boxes and confidence scores.

    :param request: Request object containing the Base64-encoded image.
    :type request: PredictRequest
    :return: Detection results.
    :rtype: PredictResponse
    :raises HTTPException 502: If TorchServe is unreachable or returns an error.
    """
    payload = json.dumps({"image": request.image})

    try:
        response = httpx.post(
            f"{TORCHSERVE_URL}/predictions/{MODEL_NAME}",
            content  = payload,
            headers  = {"Content-Type": "application/json"},
            timeout  = 30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"TorchServe error: {e}")

    result = response.json()
    return PredictResponse(detections=result.get("detections", []))


@app.post("/augment", response_model=AugmentResponse)
def augment(request: AugmentRequest):
    """Generate augmented variants of a Base64-encoded image.

    Applies the same augmentation pipeline used during training to
    produce ``n_variants`` transformed versions of the input image.
    Bounding boxes are not transformed — this endpoint is intended
    for visual inspection only.

    :param request: Request object with the image and number of variants.
    :type request: AugmentRequest
    :return: List of Base64-encoded augmented images.
    :rtype: AugmentResponse
    :raises HTTPException 400: If the Base64 image cannot be decoded.
    """
    try:
        variants = augmenter.generate_base64_variants(
            b64_image  = request.image,
            n_variants = request.n_variants,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Augmentation error: {e}")

    return AugmentResponse(variants=variants)


@app.get("/health")
def health():
    """Health check endpoint.

    :return: Service status and TorchServe reachability.
    :rtype: dict
    """
    torchserve_ok = False
    try:
        r = httpx.get(f"{TORCHSERVE_URL}/ping", timeout=5.0)
        torchserve_ok = r.status_code == 200
    except httpx.HTTPError:
        pass

    return {
        "status":     "ok",
        "torchserve": "reachable" if torchserve_ok else "unreachable",
    }