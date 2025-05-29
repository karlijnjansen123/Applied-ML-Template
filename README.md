
# Applied Machine Learning Project  
**By:** Aliesja, Francien, Ilse, and Karlijn

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
We use the HBSC (“Health Behavior in School-aged Children”) dataset from 2018, containing 244097 records of children 
across the world, aged 11, 13, and 15 years old.

## Deployment
### Requirements 
Before running the API, navigate the path to the root directory using the **'Folder Structure'** defined below, and 
install the requirements via the terminal:

`pip install  -r project_name/requirements`

### Run the API
To start the API, navigate to the root directory of the project and run in the terminal: 

`uvicorn project_name.Deployment.API:app --reload`

This will start the FastAPI server at: 

`htttp://127.0.0.1:8000`

### Send a request
There are two ways to send a request to the API:
1. Use the link defined above and put "/docs" behind the URL to view the automated documentation in the swagger UI.
2. Send a curl request via the terminal

Below there is an example or such a curl request and the corresponding response body of the API.

**Curl Request**

`curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "irritable": 3,
  "nervous": 3,
  "bodyweight": 60,
  "lifesat": 3,
  "headache": 3,
  "stomachache": 3,
  "health": 3,
  "bodyheight": 170,
  "backache": 3,
  "studyaccept": 3,
  "beenbullied": 3,
  "schoolpressure": 3,
  "talkfather": 3,
  "fastcomputers": 3,
  "dizzy": 3,
  "overweight": 0
}

**Response body**

`
{
  "Risk for body image": "Much too thin",
  "Risk at feeling low": "About every day",
  "Risk at sleep difficulties": "About every day"
}`

### Folder Structure 

The path defined below depicts all the folders, but not all the files are shown as that would make the tree unclear.

Applied-ML-Template/<br>
├── Pipfile<br> 
├── Pipfile.lock<br>
├── README.md<br>
├── __init__.py<br>
├── __pycache__<br>
│└── main.cpython-310.pyc<br> 
├── main.py<br> 
├── project_name<br> 
│   ├── Deployment<br> 
│   │  └──API.py<br>
│   ├── __init__.py<br> 
│   ├── __pycache__<br> 
│   ├── data<br> 
│   ├── features<br> 
│   ├── models<br> 
│   └── requirements<br> 
└── tests<br> 
    ├── __init__.py<br> 
    ├── data<br> 
    ├── features<br> 
    ├── models<br> 
    └── test_main.pyv

## Features

......

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
followed by four hidden layers with 128,64,32,16 units, respectively, each using ReLu activation.
The hidden layers are shared across tasks to enable multi-task learning. 
The network branches into three separate output layers, each using softmax activation to generate predictions.

#### Input and output
Sixteen input features were selected based on feature importance scores. 
The model generates predictions across the three separated domains. 

#### Training and evaluation
The features and target data are split into a train and test set.
The input data (*X_train* , *X_test*) is normalized using a *StandardScalar* to ensure consistent scaling across features. 
The neural network  is trained using the *Adam optimizer*, with *SparseCategoricalAccuracy* as the evaluation metric.
Accuracy is calculated separately for each output target to asses performance per task.

SHOW THAT THE MODEL DOES MORE THAN RANDOM GUESSING

#### Model Justification 
The multi-task learning neural network outperformed the three separate KNN-models, demonstrating the effectiveness 
of the shared representations across the three targets outputs.

#### Limitations 
There are some limitations that could affect the generalizability and performance of the model:
- The model relies on self-reported questionnaire responses which could introduce biases. 
- The target classes are imbalanced, and apart from using Focal loss no other methods were used to mitigate this. 
- Due to the imbalanced classes and possible bias due to sel-reported questionnaire answers, the models accuracy is limited.
However as stated earlier, the primary significance and novelty of this tool lies in the insight it provides of how certain 
lifestyle choices impact overall mental and physical health.
