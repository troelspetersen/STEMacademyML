import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_lgbm(X_train, y_train):
    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_xgboost(X_train, y_train):
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

st.title("Regression Model Selector")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data Loaded Successfully!")
    st.write(df.head())

    # Select target column
    target = st.selectbox("Select Target Column for Regression", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Handle categorical columns
        X = pd.get_dummies(X, drop_first=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Add buttons for regression models
        if st.button("Perform LGBM Regression"):
            # Train LGBM model
            lgbm_model = train_lgbm(X_train, y_train)

            # Predict and evaluate
            y_pred = lgbm_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"LGBM Regression - Mean Squared Error: {mse}")

        if st.button("Perform XGBoost Regression"):
            # Train XGBoost model
            xgb_model = train_xgboost(X_train, y_train)

            # Predict and evaluate
            y_pred = xgb_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"XGBoost Regression - Mean Squared Error: {mse}")
else:
    st.info("Please upload a CSV file to proceed.")
