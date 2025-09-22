import streamlit as st
st.title("Hello Streamlit")
name = st.text_input("Your name")
if name:
    st.write(f"Hello, {name} 👋")