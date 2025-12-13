import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the Model and Features
# Ensure these files are in the same folder as app.py
try:
    # Load the Random Forest model (assuming it's saved as 'random_forest_model.pkl')
    model = joblib.load('random_forest_model.pkl')
    # Load the list of feature column names (assuming it's saved as 'model_features.pkl')
    feature_columns = joblib.load('model_features.pkl')
except FileNotFoundError:
    st.error("Model files 'random_forest_model.pkl' or 'model_features.pkl' not found. Please save the model first by following Step 1.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# 2. Page Configuration and Title
st.set_page_config(
    page_title="Water Potability Predictor", 
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("Water Quality (Potability) Predictor")
st.markdown("""
This app predicts whether the water is potable (safe for drinking) or not, based on the parameters you provide.
""")

st.sidebar.header("Input Parameters")
st.markdown("---")

# 3. User Input Function (Sliders in the Sidebar)
def get_user_inputs():
    # Creating sliders for the top 5 features used in the model
    # Default values should be around the mean/median of your training data
    
    # Example ranges based on typical water quality data:
    
    Sulfate = st.sidebar.slider('Sulfate (mg/L)', 250.0, 450.0, 335.0) 
    ph = st.sidebar.slider('pH Value', 5.5, 9.5, 7.0)
    Hardness = st.sidebar.slider('Hardness (mg/L)', 100.0, 300.0, 200.0)
    Chloramines = st.sidebar.slider('Chloramines (mg/L)', 4.0, 10.0, 7.0)
    Solids = st.sidebar.slider('Solids (TDS mg/L)', 5000.0, 50000.0, 20000.0)

    # Group the features into a dictionary (maintain the order used during training)
    data = {
        'Sulfate': Sulfate,
        'ph': ph,
        'Hardness': Hardness,
        'Chloramines': Chloramines,
        'Solids': Solids,
    }
    
    # Convert the dictionary to a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get the input
input_df = get_user_inputs()

# 4. Input Display
st.subheader('Your Input Parameters')
st.dataframe(input_df, use_container_width=True)
st.markdown("---")

# 5. Prediction Logic
if st.button('Get Result: Is the Water Potable?'):
    
    # Make prediction using the model
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    
    if prediction[0] == 1:
        st.success('Prediction: **The water is likely POTABLE!**')
    else:
        st.error('Prediction: **The water is likely NOT POTABLE!**')
        
    st.markdown("---")
    st.subheader('Probabilities')
    
    prob_not_potable = prediction_proba[0][0] * 100
    prob_potable = prediction_proba[0][1] * 100
    
    st.info(f"Probability of being Not Potable (0): **{prob_not_potable:.2f}%**")
    st.info(f"Probability of being Potable (1): **{prob_potable:.2f}%**")

# Footer
st.markdown("""
<br><br>
<p style='text-align: center; color: gray;'>
This prediction is based on a Machine Learning model.
</p>
""", unsafe_allow_html=True)