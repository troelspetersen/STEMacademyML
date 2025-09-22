import streamlit as st
import pandas as pd

st.title("Data Explorer")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    col = st.selectbox("Column to plot", df.columns)
    if pd.api.types.is_numeric_dtype(df[col]):
        st.line_chart(df[col])
    else:
        st.write(df[col].value_counts())