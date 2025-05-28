"""
Start the API from the rootpath:
uvicorn project_name.Deployment.API:app --reload
"""
from fastapi import FastAPI
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

#Added so it works
@app.get("/")
async def root():
    return {"message": "API is working!!!"}

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