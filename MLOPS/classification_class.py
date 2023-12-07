import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from sklearn.base import BaseEstimator, RegressorMixin

class NeuralNetworkClassifier(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_units=(15, 19, 10), dropout_rate=0.07, activation=('relu', 'leaky_relu', 'leaky_relu'), optimizer='adam', loss='mean_squared_error', epochs=300, batch_size=35, validation_split=0.1):
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

        model.add(Dense(1, activation='sigmoid'))

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