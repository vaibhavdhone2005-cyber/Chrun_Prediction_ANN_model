import streamlit as st
import numpy as np
import pandas as pd
import keras
import pickle

# Page configuration for a professional executive look
st.set_page_config(
    page_title="Executive AI Portal | ANN Model Deployment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive dark theme, glassmorphic cards, and typography
st.markdown("""
    <style>
    /* Main Background and Text Defaults */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container Styling */
    .header-container {
        padding: 2rem 0rem 1.5rem 0rem;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 1rem;
    }
    
    /* Card Container Styling */
    .css-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Prediction Output Dashboard Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom Styling for Streamlit Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563EB 0%, #3D82F6 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1D4ED8 0%, #2563EB 100%);
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.5);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Loads the Keras 3 sequential model binary."""
    with open("ANN_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# Top Navigation Bar
st.markdown("""
    <div class="header-container">
        <div class="header-title">Artificial Neural Network Intelligence Dashboard</div>
        <div class="header-subtitle">Enterprise Machine Learning Inference Portal & Customer Analytics</div>
    </div>
""", unsafe_allow_html=True)

# Main Application Layout
col_left, col_right = st.columns([1.2, 0.8], gap="large")

with col_left:
    st.markdown("### 🎛️ Input Parameters")
    st.caption("Adjust the 10 feature values below to generate real-time neural network predictions.")
    
    if not model_loaded:
        st.error(f"Error loading `ANN_model.pkl`: {load_error}")
    
    # Grid layout for inputs (10 Features extracted from model configuration)
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        feature_1 = st.number_input("Feature 1", value=0.5, step=0.1, help="Input parameter 1")
        feature_2 = st.number_input("Feature 2", value=100.0, step=1.0, help="Input parameter 2")
        feature_3 = st.number_input("Feature 3", value=30.0, step=1.0, help="Input parameter 3")
        feature_4 = st.number_input("Feature 4", value=3.0, step=0.1, help="Input parameter 4")
        feature_5 = st.number_input("Feature 5", value=1.0, step=0.1, help="Input parameter 5")

    with f_col2:
        feature_6 = st.number_input("Feature 6", value=0.0, step=0.1, help="Input parameter 6")
        feature_7 = st.number_input("Feature 7", value=50.0, step=0.5, help="Input parameter 7")
        feature_8 = st.number_input("Feature 8", value=12.0, step=0.1, help="Input parameter 8")
        feature_9 = st.number_input("Feature 9", value=0.8, step=0.05, help="Input parameter 9")
        feature_10 = st.number_input("Feature 10", value=2.5, step=0.1, help="Input parameter 10")

    predict_btn = st.button("🚀 Run Neural Inference")

with col_right:
    st.markdown("### 📊 Inference Results")
    
    if predict_btn and model_loaded:
        # Prepare input tensor shape (1, 10)
        input_data = np.array([[
            feature_1, feature_2, feature_3, feature_4, feature_5,
            feature_6, feature_7, feature_8, feature_9, feature_10
        ]], dtype=np.float32)
        
        # Run prediction
        raw_prediction = model.predict(input_data)[0][0]
        probability_pct = raw_prediction * 100
        classification = "High Value / Positive" if raw_prediction >= 0.5 else "Low Value / Negative"
        
        # Displays Executive Card Metrics
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Probability Output</div>
                <div class="metric-value">{probability_pct:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Binary Classification Badge
        if raw_prediction >= 0.5:
            st.success(f"**Classification Outcome:** {classification}")
        else:
            st.warning(f"**Classification Outcome:** {classification}")

        # Probability Bar Visualization
        st.markdown("**Confidence Index**")
        st.progress(float(raw_prediction))

        # Model Structure Summary Overview
        with st.expander("🔍 Neural Network Architecture Details"):
            st.markdown("""
            * **Input Shape:** `(None, 10)`
            * **Dense Layer 1:** `8 Units (ReLU)`
            * **Dense Layer 2:** `7 Units (ReLU)`
            * **Output Layer:** `1 Unit (Sigmoid)`
            * **Optimizer:** `Adam (lr=0.001)`
            """)
    else:
        st.info("Adjust input feature values on the left panel and click **Run Neural Inference** to trigger evaluation.")

    # Executive Summary Card
    st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background-color: #1E293B; border-radius: 8px; border-left: 4px solid #38BDF8;">
            <p style="margin: 0; font-size: 0.85rem; color: #94A3B8;">
                <strong>Enterprise Note:</strong> Model loaded directly from serialized Keras 3 artifact using binary classification sigmoid output activation.
            </p>
        </div>
    """, unsafe_allow_html=True)
