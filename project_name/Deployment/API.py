from fastapi import FastAPI
from pydantic import BaseModel

#Create an instance of the FastAPI, this main object will handle requests
app = FastAPI()

#model = build_neural_network() --> load model using pickle

class ModelInput(BaseModel):
    """
    input converted into modelinput via the Basemodel, here we define the structure of the expected input
    /Xfeatures
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

#endpoint of the root, everytime the user send a POST request to / which has to be the path?, the function is run
@app.post('/')

#asynchronous function
async def prediction():
    #json format?
    return {'Hello': 'World'}
