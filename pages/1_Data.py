import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px

# Set Streamlit page configuration
st.set_page_config(page_title="Data", page_icon="📈")


df = pd.read_csv("assets/combined_commodity_weather.csv")


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
st.subheader('Corn Futures')
st.line_chart(df['ZC=F_Open'])

st.subheader('Corn yield by county')
df2 = pd.read_csv("assets/corn_grain.csv",
                  usecols=['YEAR', 'LOCATION', 'STATE_FIPS', 'COUNTY_FIPS',	'YIELD','AREA',	'TOTAL_YIELD'])

#user selection
year = df2['YEAR'].unique().tolist()
#predefine ranges
total_yield = df2['TOTAL_YIELD'].unique().tolist()
min_total_yield = min(total_yield)
max_total_yield = max(total_yield)
year_selected = st.multiselect('Select year', year, default=year)

#filter data based on selection
custom_yield_range = st.slider('Select total yield range', min_value = min_total_yield, max_value= max_total_yield, value =(min_total_yield, max_total_yield))

# Filter data based on the selected range
mask = (df2['YEAR'].isin(year_selected) & df2['TOTAL_YIELD'].between(*custom_yield_range))

number_of_rows = df2[mask].shape[0]
st.markdown(f'Number of rows: {number_of_rows}')

# Filter data based on the selected range
filtered_df = df2[mask]

# Display the filtered data
st.dataframe(filtered_df)

bar_chart = px.bar(filtered_df, x='LOCATION', y='YIELD', color='YIELD', title='Corn yield by county')

bar_chart.update_layout(bargap=0.5)  # Adjust padding
st.plotly_chart(bar_chart)