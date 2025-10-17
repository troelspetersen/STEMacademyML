import streamlit as st
import pandas as pd

# Cached data loading functions
@st.cache_data
def load_housing_data():
    """Load housing prices dataset with caching"""
    return pd.read_csv('data/HousingPrices_selected.csv')

@st.cache_data
def load_diabetes_data():
    """Load diabetes dataset with caching"""
    return pd.read_csv('data/diabetes_data.csv')

@st.cache_data
def load_gletsjer_data():
    """Load gletsjer dataset with caching"""
    return pd.read_csv('data/gletsjer_data_ny_0.csv')

def main():
    # Configure the page
    st.set_page_config(
        page_title="STEM Academy - Machine Learning",  # This appears in browser tab
        page_icon=":mortar_board:",  # Icon in browser tab
        layout="wide"  # Optional: use full width
    )
    
    st.title("Machine Learning - STEM Academy")

    # Welcome message
    st.write('Velkommen til Machine Learning - STEM Academy! Denne hjemmeside er et værktøj til forløbet om Machine Learning for STEM Academy. Hjemmesiden er i stand til at køre alle de gennemgåede ML-modeller online.')
    st.write('Vælg et niveau i venstre side for at for at begynde. Under hvert niveau findes tre datasæt som matcher det som står i vejledningen på PDF (Link i venstre side). Disse er vist inde på de pågældende sider så du har overblik over hvilke variable data indeholder. ' \
    'Under hvert datasæt du få lov til at køre de tilhørende modeller:')
    st.markdown(' - Huspriser: Regression og Classification  \n - Diabetes: Classification  \n - Gletsjer: Regression')
    st.write('Under "Avanceret" kan du kan også uploade dit eget datasæt og prøve at analysere det med de gennemgåede modeller. Husk du kan hente vejledningen ved at trykke på knappen i sidepanelet i venstre side. God arbejdslyst!')
    # Add a download link for guidance PDF in the sidebar
    import os
    import base64
    pdf_path = 'data/vejledning.pdf'  # Put your PDF file here
    
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Create base64 encoded PDF for download
        b64_pdf = base64.b64encode(pdf_bytes).decode()
        
        # Create a download link that looks exactly like the original
        st.sidebar.markdown(
            f"""
            <div style="margin-top: 20px;">
                <a href="data:application/pdf;base64,{b64_pdf}" download="vejledning.pdf" style="text-decoration: none; color: #555; font-size: 14px;">Hent vejledning</a>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="margin-top: 20px;">
                <span style="color: #999; font-size: 14px;">Hent vejledning (PDF ikke fundet)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    #st.write("**Navigation:** Brug sidepanelet til venstre for at navigere mellem de forskellige sider.")

if __name__ == "__main__":
    main()