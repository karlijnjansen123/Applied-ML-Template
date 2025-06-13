dict_features_change = {
    "bodyweight": "Your weight",
    "bodyheight": "Your length",
    "emcsocmed_sum": "Your social media use",
    "nervous": "How often you're feeling nervous",
    "irritable": "How often your feeling irritable",
    "lifesat": "How satisfied you are with your life right now",
    "breakfastwd": "How often you have breakfast",
    "health": "How you rated your own health",
    "fruits_2": "How many times a week you eat fruits",
    "headache": "How often your experiencing headaches",
    "fight12m": "How many physical fights you've been in in the 12 months",
    "friendcounton": (
        "If you think you have friends you can count on "
        "when things go wrong"
    ),
    "softdrinks_2": "How many times a week you drink softdrinks",
    "dizzy": "How often you're feeling dizzy",
    "sweets_2": "How often a week you're eating sweets ",
    "friendhelp": "If you think you're friends really try to help you"
}


def features_postprocessing(
    features_bodyimage,
    features_feelinglow,
    features_sleep,
):
    """
    Convert comma-separated feature keys for each output
    into their corresponding readable descriptions.

    :param features_bodyimage: comma-separated feature keys for body image
    :param features_feelinglow: " " related to feeling low
    :param features_sleep: " " related to sleep difficulties
    :return: Three lists for each output containing descriptive feature names
    """

    features_body_list = features_bodyimage.split(",")
    features_feelow_list = features_feelinglow.split(",")
    features_sleep_list = features_sleep.split(",")

    body_output = []
    feelow_output = []
    sleep_output = []

    for feature in features_body_list:
        body_output.append(dict_features_change[feature])
    for feature in features_feelow_list:
        feelow_output.append(dict_features_change[feature])
    for feature in features_sleep_list:
        sleep_output.append(dict_features_change[feature])

    return body_output, feelow_output, sleep_output


def output_model_streamlit(text):
    """
    Function to get the prediction results and important features
    from model's output, and create easy-to-read summary strings
    to show in the Streamlit app.

    :param text: model output containing predictions and top features
    :return: prediction labels and feature strings for the three outputs
    """
    predictions_dict = text["predictions"]

    # Getting the contribution features
    features_bodyimage = predictions_dict[
        "Top features attributing to body image prediction"
    ]
    features_feelinglow = predictions_dict[
        "Top features attributing to feeling low prediction"
    ]
    features_sleepdiff = predictions_dict[
        "Top features attributing to sleep prediction"
    ]

    # Getting the predictions for the output labels
    label_bodyimage = predictions_dict["Prediction for body image"]
    label_feelinglow = predictions_dict["Prediction at feeling low"]
    label_sleepdiff = predictions_dict["Prediction at sleep difficulties"]

    # Return variables to print in the streamlit application
    body_image = (
        f"Your prediction on what you think of you body image: "
        f"{label_bodyimage}, the lifestyle choices contributing to this "
        f"prediction: {features_bodyimage}."
    )
    feelinglow = (
        f"Your prediction on how often you're feeling low: "
        f"{label_feelinglow}, the lifestyle choices contributing to this "
        f"prediction: {features_feelinglow}."
    )
    sleepdiff = (
        f"Your prediction on how often you're experiencing "
        f"sleep difficulties: {label_sleepdiff}, the lifestyle choices "
        f"contributing to this prediction: {features_sleepdiff}."
    )

    return (
        label_bodyimage, label_feelinglow,
        label_sleepdiff, features_bodyimage,
        features_feelinglow, features_sleepdiff,
    )
