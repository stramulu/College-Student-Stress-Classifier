# create a streamlit app which allows the user to interact with an OpenAI chatbot to 
# use our stress random forest model to predict the stress level of the user

# to run the app, run the following command:
# streamlit run app.py

# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
import os
import joblib

# load the stress random forest model
rf = joblib.load('stress_random_forest_model.pkl')

# Streamlit app for stress prediction and OpenAI chatbot

# Function to get user input for model features
def get_user_input():
    st.header("Enter Your Information for Stress Prediction")
    # Features based on StressLevelDataset.csv columns:
    # anxiety_level,self_esteem,mental_health_history,depression,headache,blood_pressure,
    # sleep_quality,breathing_problem,noise_level,living_conditions,safety,basic_needs,
    # academic_performance,study_load,teacher_student_relationship,future_career_concerns,
    # social_support,peer_pressure,extracurricular_activities,bullying

    anxiety_level = st.slider("Anxiety Level (0-30)", 0, 30, 10)
    self_esteem = st.slider("Self-Esteem (0-30)", 0, 30, 10)
    mental_health_history = st.selectbox("Do you have a history of mental health issues?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    depression = st.slider("Depression Level (0-30)", 0, 30, 10)
    headache = st.slider("Headache Frequency (0-5)", 0, 5, 2)
    blood_pressure = st.slider("Blood Pressure (1-5, 1=Low, 5=High)", 1, 5, 3)
    sleep_quality = st.slider("Sleep Quality (1-5, 1=Poor, 5=Excellent)", 1, 5, 3)
    breathing_problem = st.slider("Breathing Problems (0-5)", 0, 5, 2)
    noise_level = st.slider("Noise Level (0-5)", 0, 5, 2)
    living_conditions = st.slider("Living Conditions (1-5, 1=Poor, 5=Excellent)", 1, 5, 3)
    safety = st.slider("Safety (1-5, 1=Unsafe, 5=Very Safe)", 1, 5, 3)
    basic_needs = st.slider("Basic Needs Met (1-5, 1=Not Met, 5=Fully Met)", 1, 5, 3)
    academic_performance = st.slider("Academic Performance (1-5, 1=Poor, 5=Excellent)", 1, 5, 3)
    study_load = st.slider("Study Load (1-5, 1=Low, 5=High)", 1, 5, 3)
    teacher_student_relationship = st.slider("Teacher-Student Relationship (1-5, 1=Poor, 5=Excellent)", 1, 5, 3)
    future_career_concerns = st.slider("Future Career Concerns (1-5, 1=None, 5=Extreme)", 1, 5, 3)
    social_support = st.slider("Social Support (1-5, 1=None, 5=Strong)", 1, 5, 3)
    peer_pressure = st.slider("Peer Pressure (1-5, 1=None, 5=Extreme)", 1, 5, 3)
    extracurricular_activities = st.slider("Extracurricular Activities (0-5)", 0, 5, 2)
    bullying = st.slider("Bullying Experience (0-5)", 0, 5, 0)

    data = {
        'anxiety_level': anxiety_level,
        'self_esteem': self_esteem,
        'mental_health_history': mental_health_history,
        'depression': depression,
        'headache': headache,
        'blood_pressure': blood_pressure,
        'sleep_quality': sleep_quality,
        'breathing_problem': breathing_problem,
        'noise_level': noise_level,
        'living_conditions': living_conditions,
        'safety': safety,
        'basic_needs': basic_needs,
        'academic_performance': academic_performance,
        'study_load': study_load,
        'teacher_student_relationship': teacher_student_relationship,
        'future_career_concerns': future_career_concerns,
        'social_support': social_support,
        'peer_pressure': peer_pressure,
        'extracurricular_activities': extracurricular_activities,
        'bullying': bullying
    }
    features = pd.DataFrame([data])
    return features

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

st.title("College Student Stress Classifier & Chatbot")

# Tabs for Prediction and Chatbot
tab1, tab2 = st.tabs(["Stress Prediction", "AI Chatbot"])

with tab1:
    user_features = get_user_input()
    if st.button("Predict Stress Level"):
        # Predict using the loaded model
        try:
            prediction = rf.predict(user_features)[0]
            stress_map = {0: "Low Stress", 1: "Moderate Stress", 2: "High Stress"}
            st.success(f"Predicted Stress Level: **{stress_map.get(prediction, prediction)}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.header("Ask the AI Chatbot about Stress, Wellbeing, or Your Results")
    if not openai_api_key:
        st.info("Please enter your OpenAI API key in the sidebar to use the chatbot.")
    else:
        # Compose a detailed system prompt with all project details except the OpenAI key
        system_prompt = (
            "You are a helpful assistant specialized in student stress, wellbeing, and interpreting stress prediction results. "
            "You are assisting users of a Streamlit web app that predicts college student stress levels using a machine learning model. "
            "Here is important background to help you answer questions accurately and empathetically:\n\n"
            "📘 About the Dataset:\n"
            "This dataset captures survey responses from 843 college students aged 18–21 regarding their experiences with stress, health, relationships, academics, and emotional well-being. "
            "The responses were collected via Google Forms using a five-point Likert scale (\"Not at all\" to \"Extremely\") and anonymized to protect privacy. "
            "It enables nuanced analysis of emotional and physical stress indicators and their correlations with academic performance and lifestyle factors.\n\n"
            "🔑 Key Features (Selected):\n"
            "👤 Demographic\n"
            "  - Gender: Coded as 0 (Male), 1 (Female)\n"
            "  - Age: Numeric age (18 to 21)\n"
            "🧠 Emotional and Stress Indicators\n"
            "  - Have you recently experienced stress in your life?\n"
            "  - Have you noticed a rapid heartbeat or palpitations?\n"
            "  - Have you been dealing with anxiety or tension recently?\n"
            "  - Do you face any sleep problems or difficulties falling asleep?\n"
            "  - Do you have trouble concentrating on your academic tasks?\n"
            "  - Have you been feeling sadness or low mood?\n"
            "  - Do you get irritated easily?\n"
            "  - Do you often feel lonely or isolated?\n"
            "🩺 Physical and Health Indicators\n"
            "  - Have you been getting headaches more often than usual?\n"
            "  - Have you been experiencing any illness or health issues?\n"
            "  - Have you gained/lost weight?\n"
            "📚 Academic & Environment Stressors\n"
            "  - Do you feel overwhelmed with your academic workload?\n"
            "  - Are you in competition with your peers, and does it affect you?\n"
            "  - Do you lack confidence in your academic performance?\n"
            "  - Do you lack confidence in your choice of academic subjects?\n"
            "  - Academic and extracurricular activities conflicting for you?\n"
            "  - Do you attend classes regularly?\n"
            "  - Are you facing any difficulties with your professors or instructors?\n"
            "  - Is your working environment unpleasant or stressful?\n"
            "  - Is your hostel or home environment causing you difficulties?\n"
            "💬 Social & Relationship Factors\n"
            "  - Do you find that your relationship often causes you stress?\n"
            "  - Do you struggle to find time for relaxation and leisure activities?\n"
            "📌 Target Variable\n"
            "  - Which type of stress do you primarily experience?: Eustress, Distress, No Stress\n\n"
            "🤖 How the Machine Learning Model Works and Evaluation:\n"
            "The app uses a Random Forest Classifier, a robust ensemble machine learning algorithm that builds multiple decision trees and combines their outputs for more accurate and stable predictions. "
            "The model was trained on the above dataset, using the students' responses to the various stress, health, academic, and social questions as input features. "
            "Each feature is either a numeric value or a coded response from the survey. "
            "The model learns patterns in these responses that are associated with different levels of stress.\n"
            "During development, we also tested a Neural Network and a Logistic Regression model for comparison. "
            "However, the Random Forest Classifier performed best overall and was selected for deployment.\n"
            "The data was split into training and test sets to evaluate performance and avoid overfitting. "
            "The Random Forest achieved the following evaluation metrics on the test set:\n"
            "  - Accuracy: 0.89\n"
            "  - Precision: 0.89\n"
            "  - Recall: 0.89\n"
            "  - F1 Score: 0.89\n"
            "  - Confusion Matrix: Shows strong correct classification for all three stress levels, with slightly more errors between moderate and high stress.\n"
            "Feature importance analysis revealed which factors most influence the prediction of stress levels. "
            "Feature importance analysis showed that blood pressure was the single most important factor in the Random Forest model. However, many other features also contributed meaningfully, highlighting that while blood pressure can be a strong indicator of stress, it is really the combination of multiple factors—emotional, academic, physical, and social—that drives stress levels overall, rather than just a select few items.\n"
            "When a user enters their information, the app processes these inputs and feeds them into the trained Random Forest model, which then predicts the user's stress level as one of three categories: "
            "0 (Low Stress), 1 (Moderate Stress), or 2 (High Stress).\n\n"
            "💡 Your Role:\n"
            "You should provide practical, empathetic advice about stress, wellbeing, and interpreting the prediction results. "
            "If the user asks about the app, explain its purpose, the dataset, and how the machine learning model works in simple terms. "
            "If the user asks about their results, help them understand what their predicted stress level means and suggest ways to manage stress. "
            "You do not have access to the user's OpenAI API key or any private information."
        )
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": system_prompt}
            ]
        user_input = st.text_input("You:", key="user_input")
        if st.button("Send", key="send_button") and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            try:
                # Use the new openai v1 API for chat completions
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.chat_history
                )
                ai_message = response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": ai_message})
            except Exception as e:
                ai_message = f"Error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": ai_message})

        # Display chat history
        for msg in st.session_state.chat_history[1:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")
