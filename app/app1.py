import streamlit as st
import joblib

# Load model, vectorizer, encoder
model = joblib.load("job_recommendation_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
labels = joblib.load("label_encoder.pkl")

st.title("🚀 JobMatch AI – Career Recommendation System")

# Multiple text inputs
company = st.text_input("Enter Company (optional)", "Google")
skills = st.text_input("Enter Skills (comma separated)", "Python, SQL, Machine Learning")
description = st.text_area("Enter Job Description (optional)", "Work on data science projects with Python and SQL")

# Combine into one string (like training)
combined_input = company + " " + skills + " " + description

if st.button("Recommend Job"):
    X_input = vectorizer.transform([combined_input])
    pred = model.predict(X_input)
    job_title = labels.inverse_transform(pred)[0]
    st.success(f"💼 Recommended Job: {job_title}")
