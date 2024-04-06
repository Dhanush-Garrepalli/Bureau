import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import json
from datetime import datetime

# Custom function to parse datetime from specific format
def custom_datetime_parser(dt_str):
    try:
        return datetime.strptime(dt_str, '%d%m%Y%H%M')
    except ValueError as e:
        print(f"DateTime parsing error: {e}")
        return pd.NaT

# Function to create a DataFrame from JSON input
def create_dataframe(json_input):
    if isinstance(json_input, dict):  # Wrap single transaction in a list
        json_input = [json_input]
    for transaction in json_input:
        if 'dateTimeTransaction' in transaction:
            transaction['dateTimeTransaction'] = custom_datetime_parser(transaction['dateTimeTransaction'])
    df_transactions = pd.DataFrame(json_input)
    for field in ['transactionAmount', 'cardBalance', 'cardholderBillingConversionRate']:
        df_transactions[field] = pd.to_numeric(df_transactions[field], errors='coerce')
    df_transactions['dateTimeTransaction'] = pd.to_datetime(df_transactions['dateTimeTransaction'], errors='coerce')
    return df_transactions

# Preprocess and encode features
def preprocess_data(df_transactions):
    encoder = LabelEncoder()
    for column in ['merchantCategoryCode', 'transactionType', 'transactionCurrencyCode', 'international', 'authorisationStatus']:
        df_transactions[f'{column}_encoded'] = encoder.fit_transform(df_transactions[column].astype(str))
    
    df_transactions['hourOfDay'] = df_transactions['dateTimeTransaction'].dt.hour
    df_transactions['dayOfWeek'] = df_transactions['dateTimeTransaction'].dt.dayofweek
    average_transaction_amount = df_transactions['transactionAmount'].mean()
    df_transactions['amountAboveAverage'] = df_transactions['transactionAmount'] > average_transaction_amount
    
    features = ['transactionAmount', 'hourOfDay', 'dayOfWeek', 'amountAboveAverage',
                'merchantCategoryCode_encoded', 'transactionType_encoded',
                'transactionCurrencyCode_encoded', 'international_encoded',
                'authorisationStatus_encoded']
    return df_transactions, features

# Randomly assign a fraud label for demonstration
def assign_fraud_label(df_transactions):
    df_transactions['isFraud'] = np.random.choice([0, 1], size=(len(df_transactions),), p=[0.95, 0.05])
    return df_transactions

# Train a simplified machine learning model
def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, n_estimators=100, max_depth=5, subsample=0.8, colsample_bytree=0.8, gamma=1, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit UI for Fraud Detection System
st.title('Fraud Detection System')

json_input = st.text_area("Input Transactions in JSON Format", '{}', height=300)

if st.button('Detect Fraud'):
    if json_input:
        try:
            input_data = json.loads(json_input)
            df_transactions = create_dataframe(input_data)
            df_transactions, features = preprocess_data(df_transactions)
            df_transactions = assign_fraud_label(df_transactions)

            if len(df_transactions) > 1:
                X = df_transactions[features]
                y = df_transactions['isFraud']
                test_size = min(0.3, (len(X) - 1) / len(X))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                minority_class_size = min(np.bincount(y_train))
                if minority_class_size > 1:
                    n_neighbors = min(5, minority_class_size - 1)
                    smote = SMOTE(random_state=42, n_neighbors=n_neighbors)
                    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                else:
                    X_train_smote, y_train_smote = X_train, y_train
                
                model = train_model(X_train_smote, y_train_smote)
                
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                classification_rep = classification_report(y_test, predictions, zero_division=0)

                st.write(f"Model Accuracy: {accuracy}")
                st.write(f"ROC AUC Score: {roc_auc}")
                st.write("Classification Report:")
                st.text(classification_rep)
            else:
                st.error("Please provide more data to enable model training and evaluation.")
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.error("Please input transaction data in JSON format.")
