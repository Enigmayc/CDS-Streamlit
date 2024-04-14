import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Main",
    page_icon="👋",
)

st.write("# It's Corn! 🌽")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    It's Corn! 🌽 is a Streamlit app that demonstrates how one can predict the rise and fall of corn's prices
    using a multi-modal approach. We have a LSTM and CNN model that takes in data from a variety of sources.

    **👈 Select a model from our sidebar!**

    ### Want to see our report?
    - Check our report [here](https://streamlit.io)

    ### Codebase
    - Check our codebase [here](https://github.com/streamlit/demo-self-driving)

    
"""
)