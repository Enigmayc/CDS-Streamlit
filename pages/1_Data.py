import streamlit as st
import time
import numpy as np
import pandas as pd

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.write("Main Table")
df = pd.read_csv("assets\combined_commodity_weather.csv")

df