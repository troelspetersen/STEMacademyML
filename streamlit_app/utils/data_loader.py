"""
Data loading utilities for the ML Academy app
"""
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes, load_wine
from .config import DATA_PATHS

@st.cache_data
def load_huspriser_dataset():
    """Load the housing prices dataset"""
    try:
        return pd.read_csv(DATA_PATHS["huspriser"])
    except FileNotFoundError:
        st.error("Huspriser datasæt ikke fundet. Tjek at filen eksisterer.")
        return pd.DataFrame()

@st.cache_data
def load_diabetes_dataset():
    """Load the diabetes dataset from sklearn"""
    diabetes = load_diabetes()
    return pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

@st.cache_data
def load_wine_dataset():
    """Load the wine dataset from sklearn (used as glacier proxy)"""
    wine = load_wine()
    return pd.DataFrame(data=wine.data, columns=wine.feature_names)

def load_datasets():
    """Load all available datasets"""
    return {
        "Huspriser": load_huspriser_dataset(),
        "Diabetes": load_diabetes_dataset(),
        "Gletsjer": load_wine_dataset()
    }

def load_user_dataset(uploaded_file):
    """Load a user-uploaded CSV file"""
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Fejl ved indlæsning af fil: {e}")
            return pd.DataFrame()
    return pd.DataFrame()