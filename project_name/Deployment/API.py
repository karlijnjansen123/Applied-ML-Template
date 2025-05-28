"""
Start the API from the rootpath:
uvicorn project_name.Deployment.API:app --reload
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import keras
from focal_loss import SparseCategoricalFocalLoss
from .prediction_postprocessing import  make_predictions, post_processing

#Create an instance of the FastAPI and load the trained model
app = FastAPI()
loaded_model = keras.models.load_model("project_name/Deployment/neural_network_model.keras", custom_objects= {"SparseCategoricalFocalLoss":SparseCategoricalFocalLoss}
)


class ModelInput(BaseModel):
    """
    Basemodel to specify our  sixteen input features
    """
    irritable: int
    nervous: int
    bodyweight: int
    lifesat: int
    headache: int
    stomachache: int
    health: int
    bodyheight: int
    backache: int
    studyaccept: int
    beenbullied: int
    schoolpressure: int
    talkfather: int
    fastcomputers: int
    dizzy: int
    overweight: int

@app.post("/predict")
#Function for predictions and creating output
async def predicition(input_data:ModelInput):
    predicitions = make_predictions(input_data,loaded_model)
    prediction_thinkbody, prediction_feelinglow, prediction_sleepdifficulties = post_processing(predicitions)

    return {
        "Risk for body image": prediction_thinkbody,
        "Risk at feeling low" : prediction_feelinglow,
        "Risk at sleep difficulties": prediction_sleepdifficulties
    }

#Handle HTTP error 404
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404 and request.url.path in ["/", "/favicon.ico"]:
        return JSONResponse(status_code=204, content=None)
    # HTTP error because of a mistyped URL
    return JSONResponse(status_code=exc.status_code, content={"detail": "This specific page was not found"})