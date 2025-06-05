"Run streamlit application by this command: python -m streamlit run project_name/Deployment/streamlit/start.py"

import streamlit as st
import json
import requests
from streamlit_postprocessing import output_model_streamlit, features_postprocessing

st.title("Let AI predict your:")
st.subheader("Sleep 💤")
st.subheader("Mental health 🧠") 
st.subheader("Body image 🪞")

st.subheader("And get to know yourself better! 📈")
st.write("This AI is trained for children between 11 and 15. This is not a diagnostic tool, but can be used to provide more insight in your lifestyle choices may influence your physical and mental welbeing")
st.write( " Use under supervision of a healthcare provider.")

st.write("  \n")
st.subheader("To make a prediction, please answer these questions first:")
st.write("What is your weight?")
Q1 = int(st.slider("kg", 0, 130, 1))
st.write("What is your height?")
Q2 = int(st.slider("cm", 0, 200, 1))
st.write("  \n")
numerical_value = 0
st.write("Please select which statements about social media were true for you this last year.")
st.write("I can’t think of anything else but the moment that I will be able to use social media again")
yes1 = st.checkbox("Agree", key="agree_q1")
if yes1:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I regularly felt dissatisfied because I wanted to spend more time on social media")
yes2 = st.checkbox("Agree", key="agree_q2")
if yes2:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I often felt bad when I could not use social media")
yes3 = st.checkbox("Agree", key="agree_q3")
if yes3:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I tried to spend less time on social media, but failed")
yes4 = st.checkbox("Agree", key="agree_q4")
if yes4:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I regularly neglected other activities (e.g hobbies, sport) because I wanted to use social media") 
yes5 = st.checkbox("Agree", key="agree_q5")
if yes5:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I regularly had arguments with others because of my social media use")
yes6 = st.checkbox("Agree", key="agree_q6")
if yes6:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I regularly lied to my parents or friends about the amount of time I spend on social media")
yes7 = st.checkbox("Agree", key="agree_q7")
if yes7:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I often used social media to escape from negative feelings")
yes8 = st.checkbox("Agree", key="agree_q8")
if yes8:
    numerical_value += 2
else:
    numerical_value += 1
st.write("  \n")
st.write("I had serious conflict with my parents, brother(s) or sister(s) because of my social media use")
yes9 = st.checkbox("Agree", key="agree_q9")
if yes9:
    numerical_value += 2
else:
    numerical_value += 1

Q3 = int(numerical_value)


q4_frequency = st.select_slider(
    "Select how often in the last 6 months you were feeling nervous:", 
    options=[
        "about every day",
        "more than once day",
        "about every week",
        "about every month",
        "rarely or never"
    ])
if q4_frequency == "about every day":
    q4 = 1
if q4_frequency == "more than once day":
    q4 = 2
if q4_frequency == "about every week":
    q4 = 3
if q4_frequency == "about every month":
    q4 = 4
if q4_frequency == "rarely or never":
    q4 = 5
Q4 = int(q4)

q5_frequency = st.select_slider(
    "Select how often in the last 6 months you were feeling irritable:", 
    options=[
        "about every day",
        "more than once day",
        "about every week",
        "about every month",
        "rarely or never"
    ])
if q5_frequency == "about every day":
    q5 = 1
if q5_frequency == "more than once day":
    q5 = 2
if q5_frequency == "about every week":
    q5 = 3
if q5_frequency == "about every month":
    q5 = 4
if q5_frequency == "rarely or never":
    q5 = 5
Q5 = int(q5)

q6_frequency = st.select_slider("Select where you feel like you are on this scale (in general)", options = ["0: the worst possible life for me", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10: the best possible life for me"])
  
if q6_frequency == "0: the worst possible life for me":
    q6 = 0
if q6_frequency == "1":
    q6 = 1
if q6_frequency == "2":
    q6 = 2
if q6_frequency == "3":
    q6 = 3
if q6_frequency == "4":
    q6 = 4
if q6_frequency == "5":
    q6 = 5
if q6_frequency == "6":
    q6 = 6
if q6_frequency == "7":
    q6 = 7
if q6_frequency == "8":
    q6 = 8
if q6_frequency == "9":
    q6 = 9
if q6_frequency == "10: the best possible life for me":
    q6 = 10
Q6 = int(q6)

q7_frequency = int(st.slider("From Monday - Friday, how many days do you have breakfast (more than a glass of milk or juice)?", 0, 5, 1))
Q7 = int(q7_frequency + 1)

q8_value = st.select_slider("In general, would you say your health is...", 
                      options=[
                          "Excellent",
                          "Good",
                          "Fair",
                          "Poor"
                      ])
if q8_value == "Excellent":
    q8 = 1
if q8_value == "Good":
    q8 = 2
if q8_value == "Fair":
    q8 = 3
if q8_value == "Poor":
    q8 = 4
Q8 = int(q8)

q9_value = st.select_slider("How often a week do you eat fruits?", 
                      options=[
                          "Never", 
                          "Less than once a week",
                          "Once a week",
                          "2-4 days a week",
                          "5-6 days a week",
                          "Once a day",
                          "More than once a day"
                      ])
if q9_value == "Never":
    q9 = 1
if q9_value == "Less than once a week":
    q9 = 2
if q9_value == "Once a week":
    q9 = 3
if q9_value == "2-4 days a week":
    q9 = 4
if q9_value == "5-6 days a week":
    q9 = 5
if q9_value == "Once a day":
    q9 = 6
if q9_value == "More than once a day":
    q9 =7

Q9 = int(q9)

q10_frequency = st.select_slider(
    "Select how often in the last 6 months you had a headache:", 
    options=[
        "about every day",
        "more than once day",
        "about every week",
        "about every month",
        "rarely or never"
    ])
if q10_frequency == "about every day":
    q10 = 1
if q10_frequency == "more than once day":
    q10 = 2
if q10_frequency == "about every week":
    q10 = 3
if q10_frequency == "about every month":
    q10 = 4
if q10_frequency == "rarely or never":
    q10 = 5
Q10 = int(q10)

q11_frequency = st.select_slider(
    "During the past 12 months, how many times were you in a physical fight?", 
    options=[
        "Never",
        "One time",
        "Twice",
        "3 times",
        "4 times or more"
    ])

if q11_frequency == "Never":
    q11 = 1
if q11_frequency == "One time":
    q11 = 2
if q11_frequency == "Twice":
    q11 = 3
if q11_frequency == "3 times":
    q11 = 4
if q11_frequency == "4 times or more":
    q11 = 5
Q11 = int(q11)

q12_value = st.select_slider(
    "How much do you agree with the statement: 'I can count on my friends when things go wrong'?", 
    options=[
        "1: Very strongly disagree",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7: Very strongly agree"
    ])
if q12_value == "1: Very strongly disagree":
    q12 = 1
if q12_value == "2":
    q12 = 2
if q12_value == "3":
    q12 = 3
if q12_value == "4":
    q12 = 4
if q12_value == "5":
    q12 = 5
if q12_value == "6":
    q12 = 6
if q12_value == "7: Very strongly agree":
    q12 = 7
Q12 = int(q12)

q13_frequency = st.select_slider(
    "How many times a week do you usually drink coke or soft drinks?", 
    options=[
        "Every day",
        "Most days",
        "About once a week",
        "Less often",
        "Never"
    ])

if q13_frequency == "Every day":
    q13 = 1
if q13_frequency == "Most days":
    q13 = 2
if q13_frequency == "About once a week":
    q13 = 3
if q13_frequency == "Less often":
    q13 = 4
if q13_frequency == "Never":
    q13 = 5
Q13 = int(q13)

q14_frequency = st.select_slider(
    "Select how often in the last 6 months you felt dizzy:", 
    options=[
        "about every day",
        "more than once day",
        "about every week",
        "about every month",
        "rarely or never"
    ])
if q14_frequency == "about every day":
    q14 = 1
if q14_frequency == "more than once day":
    q14 = 2
if q14_frequency == "about every week":
    q14 = 3
if q14_frequency == "about every month":
    q14 = 4
if q14_frequency == "rarely or never":
    q14 = 5
Q14 = int(q14)


q15_value = st.select_slider("How often a week do you eat sweets or chocolate?", 
                      options=[
                          "Never", 
                          "Less than once a week",
                          "Once a week",
                          "2-4 days a week",
                          "5-6 days a week",
                          "Once a day",
                          "More than once a day"
                      ])
if q15_value == "Never":
    q15 = 1
if q15_value == "Less than once a week":
    q15 = 2
if q15_value == "Once a week":
    q15 = 3
if q15_value == "2-4 days a week":
    q15 = 4
if q15_value == "5-6 days a week":
    q15 = 5
if q15_value == "Once a day":
    q15 = 6
if q15_value == "More than once a day":
    q15 = 7

Q15 = int(q15)

q16_value = st.select_slider(
    "How much do you agree with the statement: 'My friends really try to help me'?", 
    options=[
        "1: Very strongly disagree",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7: Very strongly agree"
    ])
if q16_value == "1: Very strongly disagree":
    q16 = 1
if q16_value == "2":
    q16 = 2
if q16_value == "3":
    q16 = 3
if q16_value == "4":
    q16 = 4
if q16_value == "5":
    q16 = 5
if q16_value == "6":
    q16 = 6
if q16_value == "7: Very strongly agree":
    q16 = 7
Q16 = int(q16)


inputs = {"bodyweight": Q1, "bodyheight": Q2, "emcsocmed_sum": Q3, "nervous": Q4, "irritable": Q5, "lifesat": Q6, "breakfastwd": Q7, "health": Q8, "fruits_2": Q9, "headache": Q10, "fight12m": Q11, "friendcounton": Q12, "softdrinks_2": Q13, "dizzy": Q14, "sweets_2": Q15, "friendhelp": Q16}

if st.button("Predict"):
    res = requests.post(url = "http://127.0.0.1:8000/predict_with_shap", data = json.dumps(inputs))
    data = res.json()
    #get the labels and the features right
    bodyimage_label, feelinglow_label, sleepdiff_label, features_body, features_feelow,features_sleep = output_model_streamlit(data)
    output_body,output_feelow,output_sleep = features_postprocessing(features_body,features_feelow,features_sleep)
    st.subheader(f"Models' response:")
    st.subheader(f"Your prediction on what you think of you body image: {bodyimage_label}")
    st.text("The factors  contributing to this prediction are:")
    st.text(f"1.{output_body[0]}\n 2.{output_body[1]},\n 3.{output_body[2]}")
    st.subheader(f"Your prediction on how often you're feeling low: {feelinglow_label}")
    st.text("The factors  contributing to this prediction are:")
    st.text(f"1.{output_feelow[0]}\n 2.{output_feelow[1]},\n 3.{output_feelow[2]}")
    st.subheader(f"Your prediction on how often you're experiencing sleep difficulties:{sleepdiff_label} ")
    st.text("The factors contributing to this prediction are:")
    st.text(f"1.{output_sleep[0]}\n 2.{output_sleep[1]},\n 3.{output_sleep[2]}")



