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
from .prediction_postprocessing import make_predictions, post_processing, postprocessing_shap
import pandas as pd
import tensorflow as tf
import shap
import numpy as np
import traceback

# Load background dataset for SHAP explainer
background = pd.read_csv('shap_background.csv').values  # (1000, num_features)

# Load your model
app = FastAPI()
model = keras.models.load_model(
    "project_name/Deployment/neural_network_model.keras",
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
)


# Create SHAP explainer once for the model
def model_predict(x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    preds = model(x_tensor, training=False)  # preds is a list of 3 arrays
    preds_np = [p.numpy() for p in preds]
    combined = np.concatenate(preds_np, axis=1)  # shape: (batch_size, total_classes)
    return combined


explainer = shap.PermutationExplainer(model_predict, background)

# Define column names (must match order of features in shap_background.csv and model input)
column_names = [
    "bodyweight", "bodyheight", "emcsocmed_sum", "nervous", "irritable", "lifesat", "breakfastwd",
    "health", "fruits_2", "headache", "fight12m", "friendcounton", "softdrinks_2", "dizzy",
    "sweets_2", "friendhelp"
]


# Function to get top 3 SHAP features per output
def get_top3_shap_features_single(explainer, X_sample, column_names):
    if X_sample.ndim == 1:
        X_sample = X_sample[np.newaxis, :]

    shap_values = explainer(X_sample)
    values = shap_values.values[0]  # For single sample, shape (total_classes, features)

    num_classes_per_output = 5
    outputs = {
        "Risk for body image": values[0:num_classes_per_output],
        "Risk at feeling low": values[num_classes_per_output:2*num_classes_per_output],
        "Risk at sleep difficulties": values[2*num_classes_per_output:3*num_classes_per_output],
    }

    top_features = {}
    for output_name, output_shap in outputs.items():
        # If shap_values.values shape is 3D (samples, total_classes, features), sum abs across classes per feature
        if len(shap_values.values.shape) == 3:
            idx = list(outputs.keys()).index(output_name)
            output_shap_values = shap_values.values[0][
                num_classes_per_output * idx: num_classes_per_output * (idx + 1), :
            ]
            feature_importance = np.sum(np.abs(output_shap_values), axis=0)
        else:
            # fallback - sum absolute values along last axis
            feature_importance = np.abs(output_shap).sum(axis=0)

        top_idx = np.argsort(feature_importance)[::-1][:3]
        top_feats = [(column_names[idx], float(feature_importance[idx])) for idx in top_idx]
        top_features[output_name] = top_feats

    return top_features


# Pydantic models
class ShapPredictionInput(BaseModel):
    features: List[float]


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

    top_features = get_top3_shap_features_single(explainer, X_input, column_names)
    features_body, features_feelinlow, features_sleep = postprocessing_shap(top_features)
    return {
        "predictions": {
            "Prediction for body image": prediction_thinkbody,
            "Prediction at feeling low": prediction_feelinglow,
            "Prediction at sleep difficulties": prediction_sleepdifficulties,
            "Top features attributing to body image prediction": features_body,
            "Top features attributing to feelinglow prediction": features_feelinlow,
            "Top features attributing to sleep prediction": features_sleep

        }}


@app.post("/predict")
async def prediction(input_data: ModelInput):
    predictions = make_predictions(input_data, model)
    (
        prediction_thinkbody,
        prediction_feelinglow,
        prediction_sleepdifficulties
    ) = post_processing(predictions)

    return {
        "Risk for body image": prediction_thinkbody,
        "Risk at feeling low": prediction_feelinglow,
        "Risk at sleep difficulties": prediction_sleepdifficulties
    }


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404 and request.url.path == "/favicon.ico":
        return Response(status_code=204)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": f"Page '{request.url.path}' not found (HTTP {exc.status_code})"}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Input data is invalid", "errors": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected internal error occurred, please try again later.",
            "traceback": tb
        }
    )
