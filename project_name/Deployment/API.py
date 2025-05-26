from fastapi import FastAPI
from pydantic import BaseModel
from project_name.models.NeuralNetwork import build_neural_network

#Create an instance of the FastAPI, this main object will handle requests
app = FastAPI()

#model = build_neural_network() --> load model using pickle

class ModelInput(BaseModel):
    """
    input converted into modelinput via the Basemodel, here we define the structure of the expected input
    /Xfeatures
    """


#endpoint of the root, everytime the user send a POST request to / which has to be the path?, the function is run
@app.post('/')
#asynchronous function
async def prediction({features: ModelInput()}):
    #json format?
    return {'reponse': 'TBT'}
