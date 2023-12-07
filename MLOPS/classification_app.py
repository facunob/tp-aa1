import streamlit as st
import numpy as np
import pandas as pd
import joblib
from classification_class import NeuralNetworkClassifier
from regression_class import input_features
# python -m streamlit run classification_app.py


st.title("Red neuronal - clasificacion")

pipe = joblib.load("classification_pipeline.joblib")


min_temp = st.slider('MinTemp', 0, 40, 15)
max_temp = st.slider('MaxTemp', 0, 40, 15)
rainfall = st.slider('Rainfall', 0, 400, 1)
windgust_speed = st.slider('WindGustSpeed', 0, 400, 44)
windspeed_3pm = st.slider('WindSpeed3pm', 0, 400, 40)
humidity_3pm = st.slider('Humidity3pm', 0, 400, 30)
pressure_3pm = st.slider('Pressure3pm', 0, 4000, 1000)
pressure_9am = st.slider('Pressure9am', 0, 400, 1000)
cloud_3pm = st.slider('Cloud3pm', 0, 400, 8)

WindGustDir_E = st.checkbox("WindGustDir_E")
WindGustDir_N = st.checkbox("WindGustDir_N")
WindGustDir_S = st.checkbox("WindGustDir_S")
WindGustDir_W = st.checkbox("WindGustDir_W")

RainToday_No = st.checkbox("RainToday_No")
RainToday_Yes = st.checkbox("RainToday_Yes")



data_to_predict = np.array([[
  min_temp, max_temp, rainfall, windgust_speed, windspeed_3pm, humidity_3pm, pressure_3pm, pressure_9am, 
  cloud_3pm, WindGustDir_E, WindGustDir_N, WindGustDir_S, WindGustDir_W, RainToday_No, RainToday_Yes
]])

prediction = pipe.predict(data_to_predict)
st.write("Lloverá" if prediction >= 0.5 else "No lloverá")
