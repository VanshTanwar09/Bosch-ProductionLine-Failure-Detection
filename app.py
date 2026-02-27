import streamlit as st
import numpy as np
from xgboost import XGBClassifier
import pandas as pd

# Load your trained model and data
@st.cache_data
def load_data():
    X = pd.read_csv('X_clean.csv', index_col=0)
    y = pd.read_csv('y_clean.csv')['Response']
    return X, y

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model('bosch_model.json')
    return model

st.set_page_config(page_title="Bosch Predictor", layout="wide")

st.title("Bosch ProductionLine Failure Predictor")
st.markdown("**AUC 0.646 | 20K parts × 200 sensors**")

# Load model
model = load_model()

# Live sensor inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**L3S29F346 (Temp)** - Range: 0.0 - 200.0")
    sensor1 = st.number_input("Reading", value=100.0, step=0.1)
    if sensor1 < 0.0 or sensor1 > 200.0:
        st.error("Select the correct range: 0.0 - 200.0")

with col2:
    st.markdown("**L3S29F347 (Pressure)** - Range: 0.0 - 200.0")
    sensor2 = st.number_input("Reading", value=75.0, step=0.1)
    if sensor2 < 0.0 or sensor2 > 200.0:
        st.error("Select the correct range: 0.0 - 200.0")

with col3:
    st.markdown("**L3S29F397 (Vibration)** - Range: 0.0 - 200.0")
    sensor3 = st.number_input("Reading", value=60.0, step=0.1)
    if sensor3 < 0.0 or sensor3 > 200.0:
        st.error("Select the correct range: 0.0 - 200.0")

if st.button("**PREDICT FAILURE RISK**", type="primary", use_container_width=True):
    # Create test data (200 columns)
    test_data = np.zeros(200)
    test_data[0] = sensor1
    test_data[1] = sensor2
    test_data[2] = sensor3

    # Predict
    risk = model.predict_proba([test_data])[0,1] * 100

    # Additional logic for high values
    if sensor1 > 150 or sensor2 > 150 or sensor3 > 150:
        risk = 80  # Force high risk for extreme values

    col1, col2 = st.columns([3,1])
    with col1:
        st.metric("**Failure Probability**", f"{risk:.2f}%")
    with col2:
        if risk > 50:
            st.error("**MAINTENANCE NOW**")
        else:
            st.success("**PRODUCTION SAFE**")

st.markdown("---")
st.caption("College Project | XGBoost AUC: 0.646")
