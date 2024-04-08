import streamlit as st
import time
import numpy as np
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(page_title="Data", page_icon="📈")


df = pd.read_csv("assets/combined_commodity_weather.csv")


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
st.subheader('Corn Futures')
st.line_chart(df['ZC=F_Open'])

