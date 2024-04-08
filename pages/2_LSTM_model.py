import streamlit as st
import time
import numpy as np
import pandas as pd
import numpy as np
import ta
from pages.LSTM import LSTMClass

st.set_page_config(page_title="LSTM", page_icon="📈")

df = pd.read_csv("assets/full_dataset.csv")

# df_ta=df[['EMA', 'RSI', 'Williams_%R', 'ZC=F_Open_Diff_absolute']]
# df_ta['ZC=F_Open_Diff_absolute'] = df_ta['ZC=F_Open_Diff_absolute'].shift(-1)
# df_ta.dropna(inplace=True)

# df_commodity=df.copy()
# df_commodity=df_commodity.drop(['Date','EMA', 'RSI', 'Williams_%R', "temperature_2m_mean","sunshine_duration",	"wind_speed_10m_max","wind_gusts_10m_max",	"wind_direction_10m_dominant",	"shortwave_radiation_sum",	"et0_fao_evapotranspiration"], axis=1)
# df_commodity['ZC=F_Open_Diff_absolute'] = df_commodity['ZC=F_Open_Diff_absolute'].shift(-1)
# df_commodity.dropna(inplace=True)

# df_weather=df.copy()
# df_weather=df_weather[["temperature_2m_mean","sunshine_duration",	"wind_speed_10m_max","wind_gusts_10m_max",	"wind_direction_10m_dominant",	"shortwave_radiation_sum",	"et0_fao_evapotranspiration",'ZC=F_Open_Diff_absolute']]
# df_weather['ZC=F_Open_Diff_absolute'] = df_weather['ZC=F_Open_Diff_absolute'].shift(-1)
# df_weather.dropna(inplace=True)

df['ZC=F_Open_Diff_absolute'] = df['ZC=F_Open_Diff_absolute'].shift(-1)
df=df.drop('Date', axis=1)
df.dropna(inplace=True)

def run_model():
    l = LSTMClass(df)
    results = l.run()

    fold_results = []

    for fold, score in enumerate(results):
        ans = {"fold": fold + 1, "accuracy": score[1]}
        fold_results.append(ans)

    df_results = pd.DataFrame(fold_results)
    return df_results

if st.button('Run Model'):
    df_results = run_model()
    st.write(df_results)