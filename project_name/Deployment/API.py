"""
Run the API from the project root:
uvicorn project_name.Deployment.API:app --reload
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import List
import keras
from focal_loss import SparseCategoricalFocalLoss
from .prediction_postprocessing import (
    make_predictions, post_processing,
    postprocessing_shap, get_top3_shap_features_single,
    explainer, column_names,
)
import numpy as np


# Load your model
app = FastAPI()
model = keras.models.load_model(
    "project_name/Deployment/neural_network_model.keras",
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
)


# Pydantic models
class ShapPredictionInput(BaseModel):
    features: List[float]


# Model input features
class ModelInput(BaseModel):
    bodyweight: int
    bodyheight: int
    emcsocmed_sum: int
    nervous: int
    irritable: int
    lifesat: int
    breakfastwd: int
    health: int
    fruits_2: int
    headache: int
    fight12m: int
    friendcounton: int
    softdrinks_2: int
    dizzy: int
    sweets_2: int
    friendhelp: int


# Predicts class labels for the 3 outcomes
# and returns top SHAP features per outcome
@app.post("/predict_with_shap")
async def predict_with_shap(input_data: ModelInput):
    X_input = np.array([
        input_data.bodyweight,
        input_data.bodyheight,
        input_data.emcsocmed_sum,
        input_data.nervous,
        input_data.irritable,
        input_data.lifesat,
        input_data.breakfastwd,
        input_data.health,
        input_data.fruits_2,
        input_data.headache,
        input_data.fight12m,
        input_data.friendcounton,
        input_data.softdrinks_2,
        input_data.dizzy,
        input_data.sweets_2,
        input_data.friendhelp
    ])

    predictions = make_predictions(input_data, model)
    (
        prediction_thinkbody,
        prediction_feelinglow,
        prediction_sleepdifficulties,
        index_class_thinkbody,
        index_class_feelinglow,
        index_class_sleep
    ) = post_processing(predictions)

    top_features = get_top3_shap_features_single(
        explainer, X_input, column_names
    )
    (
        features_body,
        features_feelinlow,
        features_sleep,
    ) = postprocessing_shap(top_features)

    return {
        "predictions": {
            "Prediction for body image": prediction_thinkbody,
            "Prediction at feeling low": prediction_feelinglow,
            "Prediction at sleep difficulties": prediction_sleepdifficulties,
            "Top features attributing to body image prediction": features_body,
            (
                "Top features attributing to feeling low "
                "prediction"
            ): features_feelinlow,
            "Top features attributing to sleep prediction": features_sleep,
        }
    }


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
):
    if exc.status_code == 404 and request.url.path == "/favicon.ico":
        return Response(status_code=204)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": (
                f"Page '{request.url.path}' not found "
                f"(HTTP {exc.status_code})"
            )
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    return JSONResponse(
        status_code=422,
        content={"detail": "Input data is invalid", "errors": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
):
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "An unexpected internal error occurred, "
                "please try again later."
            )
        }
    )
