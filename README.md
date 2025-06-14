
# Applied Machine Learning Project  
*By:* Aliesja Verheij(s4534425), Francien Vogelaar(3337235), Ilse Ten Tije(s4789997), and Karlijn Janssen(s4553357)

## Project overview
This is an AI-based API that predicts whether certain mental or physical health-related problems are present for 
children between 11-15, based on questionnaire input.
Additionally, it is also explained why the model makes this prediction based on which features most highly 
contribute to the prediction. 
The model evaluates the following problems: body image problems, feeling low and sleep difficulties. 

In order to predict specific problems, a multitask neural network receives 16 question-based features. 
These input-features are scaled and put through the pre-trained model, which outputs predictions across the three 
output-domains. The model-output is post-processed enabling easier interpretation of the output. 

This tool is designed for children aged 11-15 to use together with a parent, teacher or mental health professional. 
This is not a diagnostic tool, but used to provide insight into how lifestyle choices may influence a child's 
physical and mental wellbeing. 
The API can be integrated into a school or clinical web-interface, where it supports professionals in counseling. 
It is best used as a conversation starter, helping children reflect on their habits and how they influence their 
overall health.

## Data

The data set that is used for training the mode is the HBSC (“Health Behavior in School-aged Children”) dataset from 2018, containing 244097 records of children 
across the world, aged 11, 13, and 15 years old. It contains responses to a self-reported questionnaire.

## Deployment
Below you'll find instructions on how to use the FastAPI implementation, and the streamlit application. 

### Requirements 
This project uses venv and requirements.txt for managing the virtual environment and dependencies.<br>
It runs on Python 3.10.0.<br>
1) Install Python 3.10.0
2) Create the virtual environment by running the following line in the terminal, in the directory you want it to be:<br>
`"<path to python 3.10's python.exe>" -m venv <name of your venv>`<br>
2) For windows: activate the virtual environment by running the following line in the terminal:<br>
`.<name of your venv>\Scripts\activate.bat`<br>
   For macOS: activate the virutal environment by running the following line in the terminal:<br>
`source <name of your venv>/bin/activate`<br>
3) Install the dependencies from the root directory by running the following line in the terminal:<br>
`pip install -r project_name/requirements.txt`<br>

Before running the API or the streamlit application, navigate the path to the root directory using the **'Folder Structure'** defined below.

### Run the API
To start the API, navigate to the root directory of the project and run in the terminal: 

`uvicorn project_name.Deployment.API:app --reload`

This will start the FastAPI server at: 

`http://127.0.0.1:8000`

### Send a request
There are two ways to send a request to the API:
1. Use the URL defined above and put "/docs" behind the URL to view the automated documentation in the swagger UI.
2. Send a curl request via the terminal

Below there is an example or such a curl request and the corresponding response body of the API. Screenshots of the API
call and API documentation can be found under the directory screenshots_API

**Curl Request**

curl -X 'POST' \
  'http://127.0.0.1:8000/predict_with_shap' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "bodyweight": 60,
  "bodyheight": 170,
  "emcsocmed_sum": 13,
  "nervous": 3,
  "irritable": 2,
  "lifesat": 2,
  "breakfastwd": 4,
  "health": 2,
  "fruits_2": 5,
  "headache": 2,
  "fight12m": 2,
  "friendcounton": 3,
  "softdrinks_2": 3,
  "dizzy": 2,
  "sweets_2": 3,
  "friendhelp": 4
}'

**Response body**

{
  "predictions": {
    "Prediction for body image": "A bit too thin",
    "Prediction at feeling low": "Rarely or never",
    "Prediction at sleep difficulties": "Rarely or never",
    "Top features attributing to body image prediction": "bodyheight,irritable,emcsocmed_sum",
    "Top features attributing to feelinglow prediction": "headache,sweets_2,emcsocmed_sum",
    "Top features attributing to sleep prediction": "sweets_2,headache,fight12m"
  }
}

### Run the Streamlit demo
1) Make sure you started up the API with uvicorn
2) In the terminal run: `python -m streamlit run project_name/Deployment/streamlit/start.py`
3) Fill in some questionnaire answers and press `predict`.

### Folder Structure
The path defined below depicts all the folders.
Some information about important files can be found below.

Applied-ML-Template/<br>
├── README.md<br>
├── __init__.py<br>
├── __pycache__<br>
│   └── main.cpython-310.pyc<br>
├── group_contribution.md<br>
├── logs<br>
├── main.py<br>
├── original_79_features.txt<br>
├── project_name<br>
│   ├── Deployment<br>
│   │  └──API.py<br>
│   │  └──streamlit<br>
│   ├── __init__.py<br>
│   ├── __pycache__<br>
│   ├── data<br>
│   ├── features<br>
│   ├── models<br>
│   └── requirements.txt<br>
├── screenshots_API<br>
├── shap_background.csv <br>
└── tests<br>
    ├── __init__.py<br>
    ├── data<br>
    ├── features<br>
    ├── models<br>
    └── test_main.py<br>
**main.py**: The main file that runs all the necessary functions.<br>
**logs**: Contains all the logs for tensorboard implementation.<br> 
**original_79_features.txt**: Contains all original 79 features with corresponding column names. <br>
**shap_background.csv**: Reference dataset to compute SHAP-values.<br>
**project_name/data**:Folder containing several files for data:
- HBSC2018.csv: The used data set. <br>
- preprocessing.py: Functions for preprocessing.<br>

**project_name/Deployment**: Contains several files:<br>
- API.py: API implementation.
- /streamlit:
  - start.py: Main file for the streamlit implementation.<br>
  - streamlit_postprocessing.py: Post processing for streamlit.<br>
- prediction_postprocessing.py: Functions to get the top contributing x-features via SHAP, postprocessing for the API, prediction function for the API.<br>
- neural_network_model.keras: The file containing the saved trained neural network.<br>
- scaler.pkl: The file containing the saved scaler used for the neural network.<br>
- troubleshoot.py: <br>


**project_name/features**:Contains several files used for feature importance.<br>
- feature_correlation.py: Function that displays a heatmap showing feature correlations.<br>
- feature_importance.py: Contains functions for SHAP, feature importance. <br>

**project_name/models**: Contains several files for the different models<br>
- KNN.py contains our KNN implementation.<br>
- NeuralNetwork.py contains the function building the neural network and helper functions.<br>
- /grid_tuning: Contains trial runs for hyperparameter tuning test.<br>
- /Plots manual tuning: Plots from manual tuning. <br>
- hyperparameter_tuning.py: Functions for grid search.<br>
- manual_tuning.py: Functions for manual tuning.<br>





## Features
Our first approach was to select 5 features we thought would likely inform the outcomes, however we saw that this limited the accuracy of our model (+- 30%). Therefore, we proceeded to train the neural network on all available features (79) and keep only the most important ones. All features contain information about health and health behavior in physical, mental and social domains and can be reviewed in `original_79_features.txt`. 
The 16 most important features that inform the overall model are (in descending order):
1. bodyweight in kg. This was mainly important for predicting body image and feeling low.
2. bodyheight in cm. For all three separate outcomes this was a top-3 feature. 
3. emcsocmedsum = the sum of the scores on 9 questions about problematic social media use. 
4. nervous = how often you feel nervous
5. irritable = how often you feel irritable
6. lifesat = life satisfaction 
7. breakfastwd = how often you have breakfast on weekdays
8. health = self-reported overall health
9. fruits_2 = how often you eat fruits
10. headache = how often you have headaches
11. fight12m = how often you were in a physical fight the last 12 months
12. friendcounton = how much you feel like you can count on your friends
13. softdrinks_2 = how often you drink softdrinks
14. dizzy = how often you feel dizzy
15. sweets_2 = how often you eat sweets or chocolate
16. friendhelp = how much you feel like your friends really try to help you

## Quantification input features



| Feature       |                                                                                                                                           Question                                                                                                                                           |                                                                                                                                                                      Answer |
|:--------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| bodyweight    |                                                                                                                           How much do you weight without clothes?                                                                                                                            |                                                                                                                                                                          kg |
| bodyheight    |                                                                                                                                How tall are you without shoes                                                                                                                                |                                                                                                                                                                          cm |
| emcsocmedsum  |                                                                                                            Aggregation of the following questions: During the past year, have you                                                                                                            |                                                                                                                                                   Sum of the nine questions |
| emcsocmed_1   |                                                                                    regularly found that you can’t think of anything else but the moment that you will be able to use social media again?                                                                                     |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_2   |                                                                                                      regularly felt dissatisfied because you wanted to spend more time on social media?                                                                                                      |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_3   |                                                                                                                     often felt bad when you could not use social media?                                                                                                                      |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_4   |                                                                                                                    tried to spend less time on social media, but failed?                                                                                                                     |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_5   |                                                                                              regularly neglected other activities (e.g hobbies, sport) because you wanted to use social media?                                                                                               |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_6   |                                                                                                            _regularly had arguments with others because of your social media use?                                                                                                            |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_7   |                                                                                                regularly lied to your parents or friends about the amount of time you spend on social media?                                                                                                 |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_8   |                                                                                                                  often used social media to escape from negative feelings?                                                                                                                   |                                                                                                                                                               1= No, 2= Yes |
| emcsocmed_9   |                                                                                              had serious conflict with your parents, brother(s) or sister(s) because of your social media use?                                                                                               |                                                                                                                                                               1= No, 2= Yes |
| nervous       |                                                                                                       In the last 6 months: how often have you had the following....?  Feeling nervous                                                                                                       |                                               1 = about every day, <br/> 2 = more once/day,<br> 3 = about every week<br/>   4 = about every month<br/>  5 = rarely or never |
| irritable     |                                                                                                 In the last 6 months: how often have you had the following....?  Feeling irritable                                                                                                           |                                               1 = about every day, <br/> 2 = more once/day,<br> 3 = about every week<br/>   4 = about every month<br/>  5 = rarely or never |
| lifesat       | Here is a picture of a ladder. The top of the ladder “10” is the best possible life for you and the bottom “0” is the worst possible life for you. In general, where on the ladder do you feel you stand at the moment? Tick the box next to the number that best describes where you stand. |                                                                                                                           0(=worst possible life) - 10(=best possible life) |
| breakfastwd   |                                                                        How often do you usually have breakfast (more than a glass of milk or fruit juice)? Please tick one box for weekdays and one box for weekend.                                                                         |                                                                      1 = never <br> 2 = one day <br> 3 = two days <br> 4 = three days <br> 5 = four days <br> 6 = five days |
| health        |                                                                                                                             Would you say your health is......?                                                                                                                              |                                                                                                                       1 = excellent <br> 2 = good<br> 3 = fair<br> 4 = poor |
| fruits_2      |                                                                                                     How many times a week do you usually eat fruits? Please tick one box for each line.                                                                                                      | 1 = never <br> 2 = less than once a week <br> 3 = once a week <br> 4 = 2-4 days in the week <br> 5 = 5-6 days in the week <br> 6 = once daily <br> 7 = more than once daily |
| headache      |                                                                                                          In the last 6 months: how often have you had the following....?  Headache                                                                                                           |                                               1 = about every day, <br/> 2 = more once/day,<br> 3 = about every week<br/>   4 = about every month<br/>  5 = rarely or never |
| fight12m      |                                                                                                           During the past 12 months, how many times were you in a physical fight?                                                                                                            |                                                                                      1 = none <br> 2 = Once <br> 3 = twice <br> 4 = three times <br> 5 = four or more times |
| friendcounton |                                                                                                                        I can count on my friends when things go wrong                                                                                                                        |                                                                                                                          1(=very strongly disagree)-7(=very strongly agree) |
| softdrinks_2  |                                                                                                  How many times a week do you usually drink softdrinks? Please tick one box for each line.                                                                                                   | 1 = never <br> 2 = less than once a week <br> 3 = once a week <br> 4 = 2-4 days in the week <br> 5 = 5-6 days in the week <br> 6 = once daily <br> 7 = more than once daily |
| dizzy         |                                                                                                        In the last 6 months: how often have you had the following....?  Feeling dizzy                                                                                                        |                                               1 = about every day, <br/> 2 = more once/day,<br> 3 = about every week<br/>   4 = about every month<br/>  5 = rarely or never |
| sweets_2      |                                                                                                     How many times a week do you usually eat sweets? Please tick one box for each line.                                                                                                      | 1 = never <br> 2 = less than once a week <br> 3 = once a week <br> 4 = 2-4 days in the week <br> 5 = 5-6 days in the week <br> 6 = once daily <br> 7 = more than once daily |
| friendhelp    |                                                                                                                               My friends really try to help me                                                                                                                               |                                                                                                                          1(=very strongly disagree)-7(=very strongly agree) |

## Quantification targets
| Target         | Question                                                                                                | Answer                                                                                                                        | 
|----------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| thinkbody      | Do you think your body is........?                                                                      | 1 = Much too thin <br> 2 = A bit too thin <br> 3 = About right <br> 4 = A bit too fat <br> 5 = Much too fat                   |
| feellow        | In the last 6 months, how often have you had the following....?: Feeling low                            | 1 = about every day, <br/> 2 = more once/day,<br> 3 = about every week<br/>   4 = about every month<br/>  5 = rarely or never | 
| sleepdificulty | In the last 6 months, how often have you had the following....?: Difficulties in getting to sleep       | 1 = about every day, <br/> 2 = more once/day,<br> 3 = about every week<br/>   4 = about every month<br/>  5 = rarely or never |



## Models

### Preprocessing 
The preprocessing is applied to both the baseline and full model, and includes the following steps.

- Missing and invalid values are handles by imputing the median.
- There are nine social-media related questions which are aggregated into one variable, the *Social Media Disorder Scale* 
- All features are set to numeric types

Normalization and splitting of the data is done within the models.

### Baseline

#### Model architecture
Our baseline consists of three separate multiclass KNN implemented using scikit-learn. These separate KNN will 
generate predictions for one of the three output targets based on the input-features.


#### Input and output
The model uses the 16 health-related input features selected based on feature importance from the dataset. Every KNN has one
output target, so three in total.



#### Training and evaluation
The dataset is split into training and test sets (80/20). The input data is normalised using the *StandardScaler*, then the 
model is trained with parameter *n_neighbors = 3*. The accuracy is used as an evaluation metric, the feature importance graphs 
are implemented using its predictions. 

### Neural Network

#### Model architecture 
The model is a neural network implemented with Keras. It consists of an input layer expecting sixteen input features, 
followed by three hidden layers containing 32, 32, and 64 units, respectively, each using ReLu activation.
The hidden layers are shared across tasks to enable multitask learning. 
The network branches into three separate output layers, each using softmax activation to generate predictions.

#### Input and output
Sixteen input features were selected based on feature importance SHAP-values. 
The model generates predictions across the three separate domains. 

#### Training and evaluation
The features and target data are split into a train and test set.
The input data (*X_train* , *X_test*) is normalized using a *StandardScaler* to ensure consistent scaling across features. 
The neural network  is trained using the *Adam optimizer*, and the *SparseCategoricalFocalLoss* as loss function. 
Focal loss is used because it allows distinct class weights to each output, making it well suited for multitask learning 
situation with unbalanced classes. These weights are however, not yet implemented. 
*SparseCategoricalAccuracy* is used as an evaluation metric, the accuracy is calculated separately for each output to 
assess performance per task

#### Model Justification 
The multitask learning neural network outperformed the three separate KNN-models in F1 scores, demonstrating the effectiveness 
of the shared representations across the three targets outputs. 
To support this performance, we compared the model against a majority baseline to verify that it evaluates better than random guessing. 
This baseline predicts the most frequent class for each target variable and is a more realistic benchmark than random guessing, 
especially in presence of class imbalance. 
For each of the tree targets(Body Image, Feeling Low, Sleep Difficulty), the majority class was identified and the proportion of samples in that class was used 
as the baseline accuracy. 

The model's validation accuracy on each target was then compared to the respective majority baseline (see table). 
In all three cases, the validation accuracy exceeded the baseline, this shows that the model learned useful patterns and made better 
predictions than just picking the most common class. This performance suggests that the model works well, even with imbalanced classes.
Additionally, the comparison with the KNN baseline is also represented in the table below. 

|   | Target           | Majority Baseline Accuracy | KNN Baseline Accuracy | Validation Accuracy (Neural Network) | Above Baseline? |
|---|------------------|----------------------------|-----------------------|--------------------------------------|-----------------|
| 0 | Body Image       | 0.56                       | 0.480                 | 0.377                                | False            |
| 1 | Feeling Low      | 0.473                      | 0.458                 | 0.488                            | True            |
| 2 | Sleep Difficulty | 0.491                      | 0.412                 | 0.377                                 | False            |

Further on in the project we decided to pay special attention to the F1 score with regard to imbalanced classes, which does consistently reach a higher score in the final model compared to baseline.

#### Limitations 
There are some limitations that could affect the generalizability and performance of the model:
- The model relies on self-reported questionnaire responses which could introduce biases. The outcome is also self-reported.
- The performance is perhaps too low to make for a highly useful tool. 
However as stated earlier, the primary significance and novelty of this tool lies in the personalized insight it may provide into how certain 
lifestyle choices can impact overall mental and physical health, and can be inspirational to the user despite not always being accurate.
