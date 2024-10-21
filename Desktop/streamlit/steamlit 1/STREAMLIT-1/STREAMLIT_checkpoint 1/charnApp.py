import streamlit as st
import joblib as jb
import numpy as np

# Load the churn model
model_path = r'C:\\Users\\user\\Desktop\\streamlit\\steamlit 1\\STREAMLIT-1\\churn_model.joblib'
model = jb.load(model_path)

# Load regions from the text file
regions_path = r"C:\\Users\\user\\Desktop\\streamlit\\steamlit 1\\STREAMLIT-1\\Regions"
regions = np.loadtxt(regions_path, delimiter=',', dtype=str)

# Streamlit app title
st.title('CHURN PREDICTION APPLICATION')

# Create input fields for each feature (column)
regions_ = st.text(regions)
region = st.number_input('REGION (USE THE NUMBER)')
montant = st.number_input('MONTANT')
frequence_rech = st.number_input('FREQUENCE_RECH')
revenue = st.number_input('REVENUE')
arpu_segment = st.number_input('ARPU_SEGMENT')
frequence = st.number_input('FREQUENCE')
data_volume = st.number_input('DATA_VOLUME')

# Button to trigger prediction
if st.button('Predict'):
    try:
        # Prepare the features array
        features = np.array([[region, montant, frequence_rech, revenue, arpu_segment, frequence, data_volume]])

        # Make the prediction using the model
        prediction = model.predict(features)
        # Display the prediction result
        st.success(f'The model prediction is: {prediction[0]}')
    except Exception as e:
        st.error(f"Error: {str(e)}")

# streamlit run app.py


