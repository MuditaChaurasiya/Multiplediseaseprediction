# ========================================
# HEALTHCARE PORTAL UPGRADED - 1000+ LINES
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import random
import time

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="HEALTHCARE Portal", layout="wide")


# ---------------------------
# LOAD AI MODEL
# ---------------------------
@st.cache_data
def load_data_and_train_model():
    file_path = 'MedAI.csv'
    data = pd.read_csv(file_path, encoding='ascii')

    X = data.drop('Disease', axis=1)
    y = data['Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return X.columns, model


symptom_columns, trained_model = load_data_and_train_model()

# ---------------------------
# CSS STYLING
# ---------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #e0f7fa 0%, #fce4ec 100%); font-family: 'Arial', sans-serif;}
button, .stButton button { background-color:#4CAF50; color:white; padding:8px 16px; border:none; border-radius:8px; cursor:pointer; font-weight:bold; transition:0.3s;}
button:hover, .stButton button:hover { background-color:#45a049; transform: scale(1.05);}
.stExpander { transition: all 0.4s ease-in-out;}
.card { background-color: #ffffff; padding: 15px; margin: 10px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("🏥 HEALTHCARE Portal")
page = st.sidebar.radio("Navigate",
                        ["Home", "Disease Predictor", "Doctors", "Appointments", "Dashboard", "AI Chatbot",
                         "Feedback", "About", "Contact"])


# ---------------------------
# HOME PAGE
# ---------------------------
def home_page():
    st.title("🏥 Welcome to HEALTHCARE Portal")
    st.subheader("Your one-stop health management platform")
    st.markdown("""
    <div style='background-color:#ffebee; padding:15px; border-radius:15px; text-align:center;'>
        <h2>Stay Healthy, Stay Happy!</h2>
        <p>Explore doctors, book appointments, predict diseases, and access health tips — all in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    # Health Tips Cards
    st.markdown("### 💡 Health Tips")
    tips = [
        "Drink at least 8 glasses of water daily",
        "Exercise for 30 minutes every day",
        "Eat fruits and vegetables regularly",
        "Avoid excessive sugar and junk food",
        "Get 7-8 hours of sleep",
        "Meditate daily for mental health",
        "Take short breaks during work",
        "Keep a positive mindset",
        "Wash hands regularly",
        "Limit screen time before sleep"
    ]
    cols = st.columns(3)
    for i, tip in enumerate(tips):
        with cols[i % 3]:
            st.markdown(f"<div class='card'><h4>Tip {i + 1}</h4><p>{tip}</p></div>", unsafe_allow_html=True)


# ---------------------------
# DISEASE PREDICTOR PAGE
# ---------------------------
def disease_predictor_page():
    st.header('Interactive Disease Prediction')
    st.markdown("""
    This AI-powered app predicts potential diseases based on selected symptoms.
    This is educational and not a substitute for medical advice.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Symptoms")
        symptom_categories = {
            "General": symptom_columns[:10],
            "Respiratory": symptom_columns[10:20],
            "Gastrointestinal": symptom_columns[20:30],
            "Other": symptom_columns[30:]
        }
        symptoms = {}
        for category, category_symptoms in symptom_categories.items():
            st.markdown(f"**{category}**")
            cols = st.columns(3)
            for i, symptom in enumerate(category_symptoms):
                symptoms[symptom] = cols[i % 3].checkbox(symptom)

    with col2:
        st.subheader("Prediction")
        if st.button("Predict Disease"):
            with st.spinner("Analyzing symptoms..."):
                time.sleep(1)
            input_data = pd.DataFrame([symptoms])
            prediction = trained_model.predict(input_data)
            prediction_proba = trained_model.predict_proba(input_data)
            st.success(f"Predicted Disease: **{prediction[0]}**")

            top_3_indices = prediction_proba[0].argsort()[-3:][::-1]
            top_3_diseases = trained_model.classes_[top_3_indices]
            top_3_probabilities = prediction_proba[0][top_3_indices]

            st.markdown("### Top 3 Likely Diseases")
            for disease, prob in zip(top_3_diseases, top_3_probabilities):
                st.write(f"- {disease}: {prob:.2%}")
            st.warning("This prediction is educational only.")


# ---------------------------
# DOCTORS PAGE
# ---------------------------
def doctors_page():
    st.header("👩‍⚕️ Our Doctors")
    doctors_list = [
        {"name": "Dr. Anjali Sharma", "specialty": "Cardiologist", "experience": 12, "rating": 4.8,
         "img": "https://wallpapercave.com/wp/wp2655100.jpg"},
        {"name": "Dr. Rohit Verma", "specialty": "Neurologist", "experience": 10, "rating": 4.6,
         "img": "https://as2.ftcdn.net/jpg/06/59/35/19/1000_F_659351956_j84uErnLJU7HAlVUaxiPJ5rxmQnTqjxO.jpg"},
        {"name": "Dr. Priya Singh", "specialty": "Pediatrician", "experience": 8, "rating": 4.7,
         "img": "https://wallpapercave.com/wp/wp2655098.jpg"},
        {"name": "Dr. Karan Mehta", "specialty": "General Physician", "experience": 15, "rating": 4.9,
         "img": "https://miro.medium.com/v2/resize:fit:600/1*2PNeqoXRvTli6WRVXkxZKw.jpeg"},
        {"name": "Dr. Simran Kaur", "specialty": "Dermatologist", "experience": 9, "rating": 4.6,
         "img": "https://www.essence.com/wp-content/uploads/2017/10/1508963400/GettyImages-638647058.jpg"},
    ]

    specialty_filter = st.selectbox("Filter by Specialty",
                                    ["All", "Cardiologist", "Neurologist", "Pediatrician", "General Physician",
                                     "Dermatologist", "Orthopedist"])
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 4.0)

    for doctor in doctors_list:
        if (specialty_filter == "All" or doctor['specialty'] == specialty_filter) and doctor['rating'] >= min_rating:
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin-bottom:20px; padding:15px; border-radius:15px; box-shadow:0px 4px 12px rgba(0,0,0,0.1); background-color:#f9f9f9;'>
                <img src="{doctor['img']}" style='width:120px; height:120px; border-radius:60px; margin-right:20px;'/>
                <div>
                    <h3>{doctor['name']}</h3>
                    <p>Specialty: {doctor['specialty']}</p>
                    <p>Experience: {doctor['experience']} years</p>
                    <p>Rating: {"⭐" * int(doctor['rating'])} ({doctor['rating']})</p>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------
# APPOINTMENTS PAGE
# ---------------------------
def appointments_page():
    st.header("📅 Book Appointments")
    doctors = ["Dr. Anjali Sharma", "Dr. Rohit Verma", "Dr. Priya Singh", "Dr. Karan Mehta", "Dr. Simran Kaur",
               "Dr. Ankit Jain"]

    selected_doctor = st.selectbox("Select Doctor", doctors)
    selected_date = st.date_input("Select Date")
    selected_time = st.time_input("Select Time")

    if st.button("Book Appointment"):
        st.success(f"✅ Appointment booked with {selected_doctor} on {selected_date} at {selected_time}")
        st.balloons()


# ---------------------------
# DASHBOARD PAGE
# ---------------------------
def dashboard_page():
    st.header("📊 Health Dashboard")
    st.markdown("Insights on appointments, diseases, and symptoms")

    # Fake appointment data for charts
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    appointments_count = [random.randint(5, 20) for _ in months]
    fig1 = px.bar(x=months, y=appointments_count, labels={'x': 'Month', 'y': 'Appointments'},
                  title="Monthly Appointments")
    st.plotly_chart(fig1, use_container_width=True)

    # Disease distribution
    diseases = ["Flu", "Cold", "Diabetes", "Hypertension", "Asthma", "Migraine"]
    patients = [random.randint(10, 50) for _ in diseases]
    fig2 = px.pie(values=patients, names=diseases, title="Common Diseases")
    st.plotly_chart(fig2, use_container_width=True)

    # Symptoms frequency
    symptoms_sample = ["Fever", "Cough", "Headache", "Fatigue", "Nausea"]
    freq = [random.randint(5, 30) for _ in symptoms_sample]
    fig3 = go.Figure([go.Bar(x=symptoms_sample, y=freq)])
    fig3.update_layout(title="Symptom Frequency", xaxis_title="Symptom", yaxis_title="Count")
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------
# AI CHATBOT PAGE
# ---------------------------
def ai_chatbot_page():
    st.header("🤖 AI Health Chatbot")
    st.markdown("Ask basic health-related questions. This is educational only.")
    user_input = st.text_input("Enter your question here:")

    if st.button("Ask Chatbot"):
        response = "Sorry, I didn't understand. Please consult a doctor."
        keywords = {"fever": "Check temperature and stay hydrated",
                    "cough": "Drink warm fluids and rest",
                    "headache": "Rest and stay hydrated",
                    "flu": "Consult a doctor if symptoms worsen",
                    "pain": "Take rest and avoid strain"}
        for k in keywords:
            if k in user_input.lower():
                response = keywords[k]
        st.info(f"💬 Chatbot: {response}")


# ---------------------------
# FEEDBACK PAGE
# ---------------------------
def feedback_page():
    st.header("💬 Patient Feedback")
    name = st.text_input("Your Name")
    feedback_text = st.text_area("Your Feedback")

    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        st.balloons()
        st.markdown(f"<div class='card'><b>{name}</b>: {feedback_text}</div>", unsafe_allow_html=True)


# ---------------------------
# ABOUT PAGE
# ---------------------------
def about_page():
    st.header("ℹ️ About Us")
    st.markdown("""
    HEALTHCARE Portal is an educational demo of a professional hospital web app using Streamlit.
    Features include AI-powered disease prediction, doctor profiles, appointment booking, dashboards, AI chatbot, feedback and more.
    """)


# ---------------------------
# CONTACT PAGE
# ---------------------------
def contact_page():
    st.header("📞 Contact Us")
    st.markdown("""
    - **Email:** muditach1304@gmail.com 
    - **Phone:** +91 9074487438
    - **Address:** Satna , Madhya Pradesh, India  
    """)


# ---------------------------
# PAGE RENDERING
# ---------------------------
if page == "Home":
    home_page()
elif page == "Disease Predictor":
    disease_predictor_page()
elif page == "Doctors":
    doctors_page()
elif page == "Appointments":
    appointments_page()
elif page == "Dashboard":
    dashboard_page()
elif page == "AI Chatbot":
    ai_chatbot_page()
elif page == "Articles":
    articles_page()
elif page == "Feedback":
    feedback_page()
elif page == "About":
    about_page()
elif page == "Contact":
    contact_page()

# ---------------------------
# FOOTER
# ---------------------------
st.markdown('---')
st.caption('© 2025 HEALTHCARE Portal — Educational Demo. All rights reserved.')
