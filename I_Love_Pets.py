import streamlit as st
from utility import check_password

st.set_page_config(
    page_title = "I ♥️ Pets",
)

if not check_password():  
    st.stop()

st.title("I ♥️ Pets")

st.warning("IMPORTANT NOTICE: \n"
           "\n"
           "This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.\n"
           "\n"
           "Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output. \n"
           "\n"
           "Always consult with qualified professionals for accurate and personalized advice.", icon="⚠️")

st.sidebar.success("Select a page")
