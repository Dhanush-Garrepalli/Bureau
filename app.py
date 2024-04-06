The error you're encountering suggests there might be an issue with the way the JSON data is being processed or parsed, particularly when it's being converted into a pandas DataFrame. This can occur if the JSON structure is not as expected. To ensure robust handling of JSON input from the Streamlit UI, let's make a slight modification to how the JSON is parsed and handled, ensuring it can accommodate a variety of input formats.

Below is the revised code with an additional step to explicitly parse the JSON string using Python's built-in `json.loads()` before attempting to create a pandas DataFrame. This approach provides more control over handling the input data and error messages, making the code more resilient to different JSON structures.

```python
import streamlit as st
import pandas as pd
import numpy as np
import json  # Import the json module
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from geopy.distance import great_circle

# Function to simulate the dataset based on JSON input
def create_dataframe(json_input):
    # Convert JSON input (which is now a dict or list) directly to a DataFrame
    df_transactions = pd.DataFrame(json_input)
    return df_transactions

# Encoding and feature engineering
def preprocess_data(df_transactions):
    encoder = LabelEncoder()
    df_transactions['merchantCategoryCode_encoded'] = encoder.fit_transform(df_transactions['merchantCategoryCode'])
    df_transactions['transactionType_encoded'] = encoder.fit_transform(df_transactions['transactionType'])
    df_transactions['transactionCurrencyCode_encoded'] = encoder.fit_transform(df_transactions['transactionCurrencyCode'])
    df_transactions['international_encoded'] = encoder.fit_transform(df_transactions['international'])
    df_transactions['authorisationStatus_encoded'] = encoder.fit_transform(df_transactions['authorisationStatus'])
    
    df_transactions['hourOfDay'] = df_transactions['dateTimeTransaction'].dt.hour
    df_transactions['dayOfWeek'] = df_transactions['dateTimeTransaction'].dt.dayofweek
    average_transaction_amount = df_transactions['transactionAmount'].mean()
    df_transactions['amountAboveAverage'] = df_transactions['transactionAmount'] > average_transaction_amount
    
    features = ['transactionAmount', 'hourOfDay', 'dayOfWeek', 'amountAboveAverage',
                'merchantCategoryCode_encoded', 'transactionType_encoded',
                'transactionCurrencyCode_encoded', 'international_encoded',
                'authorisationStatus_encoded']
    return df_transactions, features

# Assuming isFraud label is randomly assigned for demonstration
def assign_fraud_label(df_transactions):
    df_transactions['isFraud'] = np.random.choice([0, 1], size=(len(df_transactions),), p=[0.95, 0.05])
    return df_transactions

# Define your machine learning model here (simplified version)
def train_model(X_train, y_train):
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Streamlit UI
st.title('Fraud Detection System')

json_input = st.text_area("Input Transactions in JSON Format", '{}', height=300)
if st.button('Detect Fraud'):
    if json_input:
        try:
            # Use json.loads() to parse the JSON string input
            input_data = json.loads(json_input)
            # Ensure input_data is a list of dictionaries
            if not isinstance(input_data, list) or not all(isinstance(item, dict) for item in input_data):
                raise ValueError("Input must be a list of JSON objects.")
                
            df_transactions = create_dataframe(input_data)
            df_transactions['dateTimeTransaction'] = pd.to_datetime(df_transactions['dateTimeTransaction'])
            df_transactions, features = preprocess_data(df_transactions)
            df_transactions = assign_fraud_label(df_transactions)

            X = df_transactions[features]
            y = df_transactions['isFraud']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # SMOTE
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            model = train_model(X_train_smote, y_train_smote)
            
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            classification_rep = classification_report(y_test, predictions, zero_division=0)

            st.write
