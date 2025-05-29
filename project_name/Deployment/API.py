"""
Run the API from the project root:
uvicorn project_name.Deployment.API:app --reload
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import keras
from focal_loss import SparseCategoricalFocalLoss
from .prediction_postprocessing import make_predictions, post_processing

# Initialize FastAPI app and load trained neural network model
app = FastAPI()
loaded_model = keras.models.load_model(
    "project_name/Deployment/neural_network_model.keras",
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss})


class ModelInput(BaseModel):
   """
   Basemodel to specify our sixteen input features for prediction
   """
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


@app.post("/predict")
# Function for predictions and creating output
async def prediction(input_data: ModelInput):
    predictions = make_predictions(input_data, loaded_model)
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


# Handle 404 error, client error
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(
        request: Request,
        exc: StarletteHTTPException
):
    if exc.status_code == 404 and request.url.path in ["/favicon.ico"]:
        # Silently ignore /favicon.ico
        return Response(status_code=204)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": (
                f"This specific page '{request.url.path}' was not found "
                f"(HTTP {exc.status_code})"
            )
        }
    )


# Handle 422, validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Input data is invalid",
            "errors": exc.errors()
        }
    )


# Handle 500, internal server error
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
