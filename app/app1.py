import os
import streamlit as st
import joblib


# Load model, vectorizer, encoder

st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This Fake News Detector is built using:
- **Python, Scikit-learn, RandomForest**
- **Streamlit for deployment**
- **Dataset: skill,company name,Job Description**

Developer: **Induri Avinash Reddy**
                
🔗 [GitHub](https://github.com/avinashreddy0)  | 
🌐 [LinkedIn](https://www.linkedin.com/in/avinash-reddy-induri-4662b832a/)
""")

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "job_recommendation_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
labels = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "job_recommendation_model.pkl"))


# ----------------------------
# UI Layout
# ----------------------------
st.set_page_config(page_title="JobMatch AI", page_icon="💼", layout="centered")

st.title("🚀 JobMatch AI – Career Recommendation System")
st.markdown(
    """
    🔎 **Find your best-fit career role instantly!**  
    Enter your details below, and our AI model will recommend the most suitable job for you.  
    Powered by **Random Forest✨**
    """
)

# ----------------------------
# Input Section
# ----------------------------
with st.form("job_input_form"):
    company = st.text_input("🏢 Company (optional)", "Google")
    skills = st.text_input("🛠️ Skills (comma separated)", "Python, SQL, Machine Learning")
    description = st.text_area("📄 Job Description (optional)", "Work on data science projects with Python and SQL")
    
    submitted = st.form_submit_button("🔮 Recommend Job")

# ----------------------------
# Prediction Section
# ----------------------------
if submitted:
    combined_input = f"{company} {skills} {description}"
    X_input = vectorizer.transform([combined_input])
    pred = model.predict(X_input)
    job_title = labels.inverse_transform(pred)[0]

    st.success(f"💼 **Recommended Job:** {job_title}")

    # Confidence / Probability (if classifier supports it)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        confidence = max(probs) * 100
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # Visual Highlight
    if "Data" in job_title or "Engineer" in job_title:
        st.markdown(f"<p style='color:limegreen; font-size:18px;'>✅ Great fit for Data roles!</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:orange; font-size:18px;'>🤔 Consider upskilling for a better fit.</p>", unsafe_allow_html=True)
