
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

To run the API:
- Start the API from the rootpath
- In your terminal run: uvicorn project_name.Deployment.API:app --reload 

To use the model deployment in a web browser make sure to put /docs behind the url.

.....

## Features

......

## Models
We use a k-Nearest Neighbors (KNN) model as the baseline, and a multi-class neural network as the primary model for classification.

......