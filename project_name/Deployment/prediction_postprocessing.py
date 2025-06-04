import numpy as np
import joblib
import pandas as pd
import tensorflow as tf
import shap
from focal_loss import SparseCategoricalFocalLoss
import keras


# Load neural network model
model = keras.models.load_model(
    "project_name/Deployment/neural_network_model.keras",
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
)


def make_predictions(input_data, loaded_model):
    """
    Function that makes predictions

    :param input_data: An instance of the ModelInput defined in API.py
    :param loaded_model: The trained model from NeuralNetwork.py
    :return: a numpy array for the outputs with softmax values for each class
    """

    user_input = np.array([
        [
            input_data.bodyweight, input_data.bodyheight,
            input_data.emcsocmed_sum, input_data.nervous,
            input_data.irritable, input_data.lifesat,
            input_data.breakfastwd, input_data.health,
            input_data.fruits_2, input_data.headache,
            input_data.fight12m, input_data.friendcounton,
            input_data.softdrinks_2, input_data.dizzy,
            input_data.sweets_2, input_data.friendhelp
        ]
    ])

    # Scale the input and predict
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

    # Softmax values for each output
    output_classes = {
        "ThinkBodyClass": int(np.argmax(prediction[0][0])),
        "FeelingLowClass": int(np.argmax(prediction[1][0])),
        "SleepDifficultiesClass": int(np.argmax(prediction[2][0]))
    }

    # Save the predictions as variables and shift to 1-based index
    thinkbody_class = output_classes["ThinkBodyClass"] + 1
    feelinglow_class = output_classes["FeelingLowClass"] + 1
    sleepdifficulties_class = output_classes["SleepDifficultiesClass"] + 1

    # Defining the values corresponding to the predicted classes
    thinkbody_dict = {
        '1': 'Much too thin',
        '2': 'A bit too thin',
        '3': 'About right',
        '4': 'A bit too fat',
        '5': 'Much too fat',
    }
    feelinglow_dict = {
        '1': 'About every day',
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

    # Save the values as variables
    index_class_thinkbody = output_classes["ThinkBodyClass"]
    index_class_feelinglow = output_classes["FeelingLowClass"]
    index_class_sleep = output_classes["SleepDifficultiesClass"]

    predicted_thinkbody = str(thinkbody_dict[str(thinkbody_class)])
    predicted_feelinglow = str(feelinglow_dict[str(feelinglow_class)])
    predicted_sleep = str(sleepdifficulties_dict[str(sleepdifficulties_class)])

    return (
        predicted_thinkbody,
        predicted_feelinglow,
        predicted_sleep,
        index_class_thinkbody,
        index_class_feelinglow,
        index_class_sleep,
    )


# Load background dataset for SHAP explainer
background = pd.read_csv('shap_background.csv').values  # (1000, num_features)


# Create SHAP explainer once for the model
def model_predict(x):
    """
    Function that converts model's multiple outputs
    into a single flat prediction array for SHAP

    :param x: Input sample
    :return: Combined (combined predictions as a 2D NumPy array)
    """

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    preds = model(x_tensor, training=False)  # preds is a list of 3 arrays
    preds_np = [p.numpy() for p in preds]
    combined = np.concatenate(preds_np, axis=1)  # shape: (batch_size, total_classes)

    return combined


# Initialize SHAP PermutationExplainer
explainer = shap.PermutationExplainer(model_predict, background)

# Define column names (must match order of features in shap_background.csv and model input)
column_names = [
    "bodyweight", "bodyheight", "emcsocmed_sum", "nervous", "irritable", "lifesat", "breakfastwd",
    "health", "fruits_2", "headache", "fight12m", "friendcounton", "softdrinks_2", "dizzy",
    "sweets_2", "friendhelp"
]


# Function to get top 3 SHAP features per output
def get_top3_shap_features_single(explainer, X_sample, column_names):
    """
    Function that computes the top 3 most influential input features for
    each model output using SHAP values

    :param explainer: SHAP Explainer instance
    :param X_sample: Input sample
    :param column_names: List of feature names in same order as model input
    :return: Dictionary mapping each output name to its top 3 features
    """

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
        # If shap_values.values shape is 3D (samples, total_classes, features)
        # then sum abs across classes per feature
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


def postprocessing_shap(top_features):
    """
    Function to convert top 3 SHAP features per output into strings for display

    :param top_features: Containing top 3 features per output category
    :return: Strings with top features for names for the 3 output targets
    """
    topfeatures_body = top_features["Risk for body image"]
    topfeatures_feelinglow = top_features["Risk at feeling low"]
    topfeatures_sleep = top_features["Risk at sleep difficulties"]
    features_body = ",".join([feature[0] for feature in topfeatures_body])
    features_feelinlow = ",".join([feature[0] for feature in topfeatures_feelinglow])
    features_sleep = ",".join([feature[0] for feature in topfeatures_sleep])

    return features_body, features_feelinlow, features_sleep
