"""
Start the API from the rootpath:
uvicorn project_name.Deployment.API:app --reload
"""

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
#Added to get it to work?
@app.get("/")
async def root():
    return {"message": "API is working!!!"}

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
    #Ouput with just the predictions can be deleted?

    output = {
        "ThinkBody": prediction[0][0].tolist(),
        "FeelingLow": prediction[1][0].tolist(),
        "Sleep Difficulties" : prediction[2][0].tolist(),
    }

    output_classes ={
        "ThinkBodyClass": int(np.argmax(prediction[0][0]))+1,
        "FeelingLowClass": int(np.argmax(prediction[1][0]))+1,
        "SleepDifficultiesClass": int(np.argmax(prediction[2][0])+1)
    }

    #Classes for the output
    thinkbody_class = output_classes["ThinkBodyClass"]
    feelinglow_class = output_classes["FeelingLowClass"]
    sleepdifficulties_class = output_classes["SleepDifficultiesClass"]

    #Defining the classes dictionaries:
    thinkbody_dict = {
        '1':'Much too thin',
        '2': 'A bit too thin',
        '3': 'About right',
        '4': 'A bit too fat',
        '5': 'Much too fat',
    }
    feelinglow_dict = {
        '1':'About every day',
        '2': 'About once a week',
        '3': 'About every week',
        '4': 'About every month',
        '5': 'Rarely or never'
    }
    sleepdifficulties_dict = {
        '1': 'About every day',
        '2': 'About once a week',
        '3': 'About every week',
        '4': 'About every month',
        '5': 'Rarely or never'
    }
    output_formatted = {
        "Risk for body image" : str(thinkbody_dict[str(thinkbody_class)]),
        "Risk at feeling low" : str(feelinglow_dict[str(feelinglow_class)]),
        "Risk at sleep difficulties" : str(sleepdifficulties_dict[str(sleepdifficulties_class)])
    }

    print(f"You're at risk to think the following about your body: {thinkbody_dict[str(thinkbody_class)]}")
    print(f"You're at risk to feel low {feelinglow_dict[str(feelinglow_class)]}")
    print(f"You're at risk to have difficulties with sleep {sleepdifficulties_dict[str(sleepdifficulties_class)]}")

    print(output_classes)
    return output_formatted
