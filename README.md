
# Applied Machine Learning Project  
**By:** Aliesja, Francien, Ilse, and Karlijn

## Project overview
This is an AI-based API that predicts whether certain mental or physical health-related problems are present for children between 11-15, based on questionnaire input.
Additionally, it is also explained why the model makes this prediction based on which features most highly contribute to the prediction. 
The model evaluates problems: body image problems, feeling low and sleep difficulties. 

In order to predict specific problems, a multitask neural network receives 16 question-based features. These input-features are scaled 
and put through the pre-trained model, which outputs predictions across the three output-domains. The model-output is post-processed
enabling easier interpretation of the output. 

This tool is designed for children aged 11-15 to use together with a parent, teacher or mental health professional. 
This is not a diagnostic tool, but used to provide insight into how lifestyle choices may influence a child's 
physical and mental wellbeing. 
The API can be integrated into a school or clinical web-interface, where it supports professionals in counseling. It is best 
used as a conversation starter, helping children reflect on their habits and how they influence their overall health.

## Data
We use the HBSC (“Health Behavior in School-aged Children”) dataset from 2018, containing 244097 records of children across the world, aged 11, 13, and 15 years old.  
....

## Deployment
### Requirements 
Before running the API, navigate the path to the root directory using the **'Folder Structure'** defined below, and install the requirements via the terminal:

`pip install -r requirements.txt`

### Run the API
To start the API, navigate to the root directory of the project and run in the terminal:

`uvicorn project_name.Deployment.API:app --reload`

This will start the FastAPI server at:

[FastAPI server](htttp://127.0.0.1:8000)

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

**Response body**<br>
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


.....

## Features

......

## Models
### Baseline
We use a k-Nearest Neighbors (KNN) model as the baseline, and a multi-class neural network as the primary model for classification.
### Full model 
......