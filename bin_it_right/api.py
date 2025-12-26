import os
import io
from typing import Annotated, Dict
from fastapi import FastAPI, File, UploadFile
from pydantic import Field, BaseModel
from bin_it_right.modeling.classifier import TrashClassifier, ClassificationResponse
from .config import DATA_PATH
from PIL import Image

api = FastAPI(
    title="Bin it right API",
    description="Application for trash classification",
    version="1.0.0",
)

class PredictionResponse(BaseModel):
    pred_class: str
    classes: Dict[str, float]



@api.post("/{model_type}/predict", response_model=PredictionResponse)
async def preduct(model_type: str, file: UploadFile = File(...) ) -> PredictionResponse:
    object_content = await file.read()
    image = Image.open(io.BytesIO(object_content))
    classifier = TrashClassifier(
        base_path=os.path.join(DATA_PATH, "processed")
    )
    response = classifier.classify(
        model_type=model_type,
        image=image
    )
    return PredictionResponse(
        pred_class=response.predicted_class,
        classes=response.classes_distribution
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "bin_it_right.api:api",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        reload=bool(os.environ.get('RELOAD', False)),
    )