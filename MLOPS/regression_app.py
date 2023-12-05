import streamlit as st
import numpy as np
import pandas as pd
import joblib
from regression_class import feat_eng, NeuralNetworkRegressor
# python -m streamlit run regression_app.py


input_features = [
  'MinTemp', 'MaxTemp', 'Rainfall',  'WindGustSpeed', 'WindSpeed3pm',
  'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud3pm',
  'WindGustDir_E', 'WindGustDir_N', 'WindGustDir_S', 'WindGustDir_W',
  'RainToday_No', 'RainToday_Yes',
]

st.title("Red neuronal - prediccion de lluvia")

pipe = joblib.load(r"C:\Users\facuf\Downloads\regression_pipeline.joblib")


# min_temp = st.slider('MinTemp', 0, 40, 15)
# max_temp = st.slider('MaxTemp', 0, 40, 15)
# rainfall = st.slider('Rainfall', 0, 400, 1)
# windgust_speed = st.slider('WindGustSpeed', 0, 400, 44)
# windspeed_3pm = st.slider('WindSpeed3pm', 0, 400, 40)
# humidity_3pm = st.slider('Humidity3pm', 0, 400, 30)
# pressure_3pm = st.slider('Pressure3pm', 0, 4000, 1000)
# pressure_9am = st.slider('Pressure9am', 0, 400, 1000)
# cloud_3pm = st.slider('Cloud3pm', 0, 400, 8)

# WindGustDir_E = st.checkbox("WindGustDir_E")
# WindGustDir_N = st.checkbox("WindGustDir_N")
# WindGustDir_S = st.checkbox("WindGustDir_S")
# WindGustDir_W = st.checkbox("WindGustDir_W")

# RainToday_No = st.checkbox("RainToday_No")
# RainToday_Yes = st.checkbox("RainToday_Yes")



# data_to_predict = np.array([[
#   min_temp, max_temp, rainfall, windgust_speed, windspeed_3pm, humidity_3pm, pressure_3pm, pressure_9am, 
#   cloud_3pm, WindGustDir_E, WindGustDir_N, WindGustDir_S, WindGustDir_W, RainToday_No, RainToday_Yes
# ]])

# prediction = pipe.predict(data_to_predict)

def get_user_input():
    input_dict = {}

    with st.form(key='my_form'):
        for feat in input_features:
            input_value = st.number_input(f"Enter value for {feat}", value=0.0, step=0.01)
            input_dict[feat] = input_value

       
        submit_button = st.form_submit_button(label='Submit')

    return pd.DataFrame([input_dict]), submit_button


user_input, submit_button = get_user_input()


if submit_button:
    prediction = pipe.predict(user_input)
    prediction_value = prediction[0]

    st.header("Predicted Quality")
    st.write(prediction_value)
