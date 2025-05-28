"""
Start the API from the rootpath:
uvicorn project_name.Deployment.API:app --reload
"""

from fastapi import FastAPI
from numpy.ma.core import argmax
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
#Added to get it to work?
@app.get("/")
async def root():
    return {"message": "✅ The API is running. Use /predict to get predictions."}
#Define a route (what comes after the url
@app.post("/predict")
#The parameter for this asynchronous function is an instance of the ModelInput
async def make_predicition(input_data:ModelInput):
    user_input = np.array([[input_data.irritable, input_data.nervous,
                input_data.bodyweight,input_data.lifesat,input_data.headache,input_data.stomachache,
                input_data.health, input_data.bodyheight,input_data.backache,input_data.studyaccept,input_data.beenbullied,
                input_data.schoolpressure,input_data.talkfather,input_data.fastcomputers,input_data.dizzy,input_data.overweight]])
    #Predict, returns a  list of numpy arrays, one for each output
    prediction  = loaded_model.predict(user_input)

    #Formatting the output to JSON format (from numpy to a python list), and stored in a dictionary
    #Also encoding the class, np.argmax returns the index of the highest value
    output = {
        "ThinkBody": prediction[0][0].tolist(),
        "ThinkBodyClass": int(np.argmax(prediction[0][0])),
        "FeelingLow": prediction[1][0].tolist(),
        "FeelingLowClass":int(np.argmax(prediction[1][0])),
        "Sleep Difficulties" : prediction[2][0].tolist(),
        "SleepDifficultiesClass" :int(np.argmax(prediction[2][0]))}
    print(output)
    return output
