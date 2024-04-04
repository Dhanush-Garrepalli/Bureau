import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load('random_forest_model.pkl')

# Streamlit app title
st.title('Fraud Detection Tool')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # Assuming your model expects the same column names as in the CSV
    # and the CSV does not contain a target column
    predictions = model.predict(input_df)

    # Add predictions to the dataframe
    input_df['Prediction'] = predictions

    # Filter out the fraudulent transactions
    fraudulent_df = input_df[input_df['Prediction'] == 1]

    if not fraudulent_df.empty:
        st.write("Fraudulent Transactions Detected:")
        st.write(fraudulent_df)
    else:
        st.write("No fraudulent transactions detected.")
