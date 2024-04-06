import streamlit as st
import pandas as pd
import numpy as np
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
            input_data = pd.read_json(json_input, orient='records')
            input_data['dateTimeTransaction'] = pd.to_datetime(input_data['dateTimeTransaction'])
            df_transactions = create_dataframe(input_data)
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

            st.write(f"Model Accuracy: {accuracy}")
            st.write(f"ROC AUC Score: {roc_auc}")
            st.write("Classification Report:")
            st.text(classification_rep)
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.error("Please input transaction data in JSON format.")

