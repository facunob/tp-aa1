
import pandas as pd

ubicaciones = ['Sydney', 'SydneyAirport', 'Canberra', 'Melbourne', 'MelbourneAirport']

seasons = { 1: "verano", 2: "otoño", 3: "invierno", 4: "primavera"}
def get_season(date):
  year = date.year
  start_dates = [(year, 3, 20), (year, 6, 20), (year, 9, 21), (year, 12, 21)]

  for i in range(len(start_dates)):
    if date < pd.Timestamp(*start_dates[i]):
      return seasons.get(i+1)
  return seasons.get(1)

def get_wind(wind_dir):
  return wind_dir[0:1]


def feat_eng(df):
  df = df[df['Location'].isin(ubicaciones)]
  df.drop(['Location'], axis=1, inplace=True)

  df = df[df['RainToday'].notna()]
  df = df[df['RainTomorrow'].notna()]
  df = df[df['RainfallTomorrow'].notna()]

  df = df[df['WindGustDir'].notna()]
  df = df[df['WindDir3pm'].notna()]
  df = df[df['WindDir9am'].notna()]

  df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
  df["Season"] = df["Date"].apply(get_season)

  df["WindGustDir"] = df["WindGustDir"].apply(get_wind)
  df["WindDir3pm"] = df["WindDir3pm"].apply(get_wind)
  df["WindDir9am"] = df["WindDir9am"].apply(get_wind)

  df = pd.get_dummies(df)

  return df[[
              'MinTemp', 'MaxTemp', 'Rainfall',  'WindGustSpeed', 'WindSpeed3pm',
              'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud3pm',
              'WindGustDir_E', 'WindGustDir_N', 'WindGustDir_S', 'WindGustDir_W',
              'RainToday_No', 'RainToday_Yes',
            ]]




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.base import BaseEstimator, RegressorMixin

class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_units=(15, 18, 9), dropout_rate=0.063, activation=('relu', 'leaky_relu', 'leaky_relu'), optimizer='adam', loss='mean_squared_error', epochs=300, batch_size=35, validation_split=0.1):
        self.input_dim = 15
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_units[0], activation=self.activation[0], input_dim=self.input_dim))
        model.add(Dropout(self.dropout_rate))
        
        for units, act in zip(self.hidden_units[1:], self.activation[1:]):
            model.add(Dense(units, activation=act))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mse'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return K.eval(tf.keras.metrics.mean_squared_error(y, y_pred))