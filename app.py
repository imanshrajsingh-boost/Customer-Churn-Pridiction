import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# 2. App Title & Subtitle
st.title("🔮 Customer Churn Prediction App")
st.markdown("This app predicts whether a customer is likely to churn (leave the business) based on their profile and usage patterns.")
st.markdown("---")

# 3. Artifacts Load Function using 'requests' with SSL bypass
@st.cache_resource
def load_models():
    model_path = 'Best_Model.pkl'
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading full-accuracy model from secure cloud (~364MB)... Please wait, this takes 1-2 minutes."):
            # Google Drive Direct Download URL
            url = "https://drive.google.com/uc?export=download&id=1EnlOxm9as7A7Yt1jte5wFbGiC1Ug8-xi"
            
            # Requests use karke verfication=False lagaya taaki Error 60 bypass ho sake
            response = requests.get(url, verify=False, stream=True)
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    model = joblib.load(model_path)
    scaler = joblib.load('Scaler.pkl')
    ohe = joblib.load('One_Hot_Encoder.pkl')
    oe = joblib.load('Ordinal_Encoder.pkl')
    return model, scaler, ohe, oe

try:
    model, scaler, ohe, oe = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'Scaler.pkl', 'One_Hot_Encoder.pkl', and 'Ordinal_Encoder.pkl' are in your GitHub repo.")
    st.stop()

# 4. SIDEBAR - User Input Form
st.sidebar.header("📋 Customer Information")

with st.sidebar.form(key='customer_form'):
    # Numerical Inputs
    age = st.sidebar.slider("Age", min_value=18, max_value=65, value=30)
    tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=60, value=12)
    usage_freq = st.sidebar.number_input("Usage Frequency (per month)", min_value=1, max_value=30, value=15)
    support_calls = st.sidebar.number_input("Support Calls", min_value=0, max_value=10, value=2)
    payment_delay = st.sidebar.number_input("Payment Delay (Days)", min_value=0, max_value=30, value=5)
    total_spend = st.sidebar.number_input("Total Spend ($)", min_value=100, max_value=1000, value=500)
    last_interaction = st.sidebar.slider("Last Interaction (Days ago)", min_value=1, max_value=30, value=10)
    
    st.sidebar.markdown("---")
    
    # Categorical Inputs
    gender = st.sidebar.selectbox("Gender", options=['Male', 'Female'])
    subscription_type = st.sidebar.selectbox("Subscription Type", options=['Basic', 'Standard', 'Premium'])
    contract_length = st.sidebar.selectbox("Contract Length", options=['Monthly', 'Quarterly', 'Annual'])
    
    # Submit Button
    submit_button = st.form_submit_button(label='🔮 Predict Churn')

# 5. MAIN PAGE - Prediction Logic & Display
if submit_button:
    input_data = {
        'Age': float(age),
        'Gender': gender,
        'Tenure': float(tenure),
        'Usage Frequency': float(usage_freq),
        'Support Calls': float(support_calls),
        'Payment Delay': float(payment_delay),
        'Subscription Type': subscription_type,
        'Contract Length': contract_length,
        'Total Spend': float(total_spend),
        'Last Interaction': float(last_interaction)
    }
    
    df = pd.DataFrame([input_data])
    
    # --- PREPROCESSING ---
    num_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    ordinal_cols = ['Subscription Type', 'Contract Length']
    onehot_cols = ['Gender']
    
    df[num_cols] = scaler.transform(df[num_cols])
    
    ohe_df = pd.DataFrame(ohe.transform(df[onehot_cols]), columns=ohe.get_feature_names_out(onehot_cols), index=df.index)
    df = df.drop(columns=onehot_cols)
    df = pd.concat([df, ohe_df], axis=1)
    
    df[ordinal_cols] = oe.transform(df[ordinal_cols])
    
    # --- PREDICTION ---
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]
    
    # --- DISPLAY RESULTS ---
    st.subheader("📊 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1.0:
            st.error("⚠️ **High Risk of Churn!**")
            st.markdown("The model predicts that this customer is likely to **leave** the business.")
        else:
            st.success("✅ **Low Risk (Loyal Customer)**")
            st.markdown("The model predicts that this customer is likely to **stay**.")
            
    with col2:
        st.write("**Confidence Metrics:**")
        st.write(f"Stay Probability: {prediction_proba[0]*100:.1f}%")
        st.progress(float(prediction_proba[0]))
        
        st.write(f"Churn Probability: {prediction_proba[1]*100:.1f}%")
        st.progress(float(prediction_proba[1]))

else:
    st.info("← Please enter the customer details in the sidebar and click on **Predict Churn** to see the analysis.")
