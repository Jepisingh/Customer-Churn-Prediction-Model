import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# 1. Load the Model and Scaler (Cached for performance)
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('churn_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_resources()

# 2. Custom CSS for Black/Dark Theme
st.markdown("""
    <style>
    /* Main Background - Black */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Title Style */
    .title-text {
        color: #ffffff;
        font-family: 'Helvetica', sans-serif;
        text-align: center;
        font-weight: bold;
        padding-bottom: 20px;
    }

    /* General Text Color (ensure all text is white) */
    p, label, h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Input Container Styling (Dark Grey to contrast with Black) */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent; 
    }
    
    /* Make input labels and values visible */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #ffffff !important;
    }

    /* Button Style (Bright Blue for contrast) */
    div.stButton > button {
        background-color: #29b6f6; 
        color: black;
        width: 100%;
        padding: 10px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #039be5;
        color: white;
    }

    /* Result Box Style - Dark Mode Friendly */
    .result-box-high {
        background-color: #4a0d0d; /* Dark Red Background */
        color: #ffcdd2; /* Light Red Text */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #ef5350;
        margin-top: 20px;
    }
    .result-box-low {
        background-color: #0d3312; /* Dark Green Background */
        color: #c8e6c9; /* Light Green Text */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #66bb6a;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. App Layout
st.markdown('<h1 class="title-text">Customer Churn Prediction Model</h1>', unsafe_allow_html=True)
st.write("Enter the customer's details below to predict the probability of them leaving the bank.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=75000.0)

# 4. Prediction Logic
if st.button("Predict Churn Risk"):
    # -- Preprocessing Steps --
    
    # Encode Categorical Variables manually to match model training
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0
    
    # Binary variables
    has_card_int = 1 if has_cr_card == "Yes" else 0
    is_active_int = 1 if is_active_member == "Yes" else 0

    # Create a DataFrame with columns in the EXACT order the Scaler expects
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_card_int],
        'IsActiveMember': [is_active_int],
        'EstimatedSalary': [estimated_salary],
        'Geography_Germany': [geo_germany],
        'Geography_Spain': [geo_spain],
        'Gender_Male': [gender_male]
    })

    # Scale the data using the loaded scaler
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)
    probability = prediction[0][0]

    # 5. Display Result
    st.subheader("Prediction Result")
    
    if probability > 0.5:
        st.markdown(f"""
            <div class="result-box-high">
                <h2>⚠️ High Churn Risk</h2>
                <h3>{(probability * 100):.2f}% Probability</h3>
                <p>This customer is likely to leave the bank.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-box-low">
                <h2>✅ Low Churn Risk</h2>
                <h3>{(probability * 100):.2f}% Probability</h3>
                <p>This customer is likely to stay.</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Powered by TensorFlow & Streamlit")