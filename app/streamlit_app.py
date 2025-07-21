import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# Load model
model_path = os.path.join('models', 'xgb_model.pkl')
model = joblib.load(model_path)

# Streamlit UI config
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.write("Enter applicant details to check their credit risk.")

# Sidebar info
with st.sidebar:
    st.title("📌 App Info")
    st.write("🔍 Model: XGBoost Classifier")
    st.write("📊 Accuracy: 77% (on test set)")
    st.write("📁 Trained on: German Credit Dataset")

# User Inputs
age = st.slider("Age", 18, 75, 30)
sex = st.selectbox("Sex", options=["male", "female"])
job = st.selectbox("Job Type", [0, 1, 2, 3])
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_account = st.selectbox("Saving Account", ["little", "moderate", "quite rich", "rich", "unknown"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=20000, value=1000)
duration = st.slider("Loan Duration (months)", 4, 72, 12)
purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "vacation/others"])

# Manual encoding
def encode_input():
    housing_map = {"own": 2, "free": 0, "rent": 1}
    saving_map = {"little": 1, "moderate": 2, "quite rich": 3, "rich": 4, "unknown": 0}
    checking_map = {"little": 1, "moderate": 2, "rich": 3, "unknown": 0}
    purpose_map = {
        "radio/TV": 0, "education": 1, "furniture/equipment": 2,
        "car": 3, "business": 4, "vacation/others": 5
    }

    return pd.DataFrame([{
        "Age": age,
        "Sex": 1 if sex == "male" else 0,
        "Job": job,
        "Housing": housing_map[housing],
        "Saving accounts": saving_map[saving_account],
        "Checking account": checking_map[checking_account],
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose_map[purpose]
    }])

# Predict and Display
# Predict and store in session state
if st.button("Predict Credit Risk"):
    input_df = encode_input()
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = proba[prediction] * 100

    # Save to session state
    st.session_state.predicted = True
    st.session_state.input_df = input_df
    st.session_state.prediction = prediction
    st.session_state.confidence = confidence

# Show prediction results only if prediction is done
if st.session_state.get("predicted", False):
    input_df = st.session_state.input_df
    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    st.subheader("Applicant Summary")
    st.write(input_df)

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("✅ Good Credit Risk")
    else:
        st.error("❌ Bad Credit Risk")
    st.write(f"🔎 Model Confidence: **{confidence:.2f}%**")

    # Feature Importance checkbox
    st.subheader("📊 Feature Importance (Top Predictive Features)")
    if st.checkbox("Show Feature Importance Chart"):
        image_path = os.path.join("images", "feature_importance.png")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption="Model Feature Importance", use_container_width=True)
        else:
            st.warning("Feature importance image not found. Please generate it using feature_importance.py.")

    # Confusion Matrix checkbox
    st.subheader("📊 Confusion Matrix")
    if st.checkbox("Show Confusion Matrix"):
        conf_image_path = os.path.join("images", "confusion_matrix.png")
        if os.path.exists(conf_image_path):
            image = Image.open(conf_image_path)
            st.image(image, caption="Confusion Matrix", use_container_width=True)
        else:
            st.warning("Confusion matrix image not found. Please generate it using train_models.py.")
