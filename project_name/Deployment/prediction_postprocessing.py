import numpy as np
import joblib

def make_predictions (input_data,loaded_model):
    """
    Function that makes predictions
    :param input_data: An instance of the ModelInput defined in API.py
    :param loaded_model: The trained model from NeuralNetwork.py
    :return: a numpy array for the outputs with softmax values for each class
    """
    user_input = np.array([[input_data.irritable, input_data.nervous,
                            input_data.bodyweight, input_data.lifesat, input_data.headache, input_data.stomachache,
                            input_data.health, input_data.bodyheight, input_data.backache, input_data.studyaccept,
                            input_data.beenbullied,
                            input_data.schoolpressure, input_data.talkfather, input_data.fastcomputers,
                            input_data.dizzy, input_data.overweight]])


    #Scale the input and predict
    scaler = joblib.load("project_name/Deployment/scaler.pkl")
    scaled_input = scaler.transform(user_input)
    prediction = loaded_model.predict(scaled_input)
    return prediction

def post_processing(prediction):
    """
    Function that finds the right predicted class from the softmax value's
    :param prediction: output of the predictions
    :return: 3 integers with the right class per output
    """
    #Softmax values for each output
    output_classes ={
        "ThinkBodyClass": int(np.argmax(prediction[0][0]))+1,
        "FeelingLowClass": int(np.argmax(prediction[1][0]))+1,
        "SleepDifficultiesClass": int(np.argmax(prediction[2][0])+1)
    }

    #Sace them as variables
    thinkbody_class = output_classes["ThinkBodyClass"]
    feelinglow_class = output_classes["FeelingLowClass"]
    sleepdifficulties_class = output_classes["SleepDifficultiesClass"]

    #Defining the values corresponding to the predicted classes
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
    #save the values as variables 
    predicted_thinkbody=str(thinkbody_dict[str(thinkbody_class)])
    predicted_feelinglow = str(feelinglow_dict[str(feelinglow_class)])
    predicted_sleep = str(sleepdifficulties_dict[str(sleepdifficulties_class)])
    return predicted_thinkbody, predicted_feelinglow, predicted_sleep

