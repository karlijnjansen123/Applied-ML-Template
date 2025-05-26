from fastapi import FastAPI
from pydantic import BaseModel
import numpy as  np
import keras
from focal_loss import SparseCategoricalFocalLoss

#Create an instance of the FastAPI, this main object will handle requests
app = FastAPI()
#load the model from the saved model.keras file,
loaded_model = keras.models.load_model("project_name/Deployment/neural_network_model.keras", custom_objects= {"SparseCategoricalFocalLoss":SparseCategoricalFocalLoss}
)

class ModelInput(BaseModel):
    """
    Basemodel to specify our input features
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