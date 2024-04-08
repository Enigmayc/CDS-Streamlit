import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras import backend as K

# Assuming self.dataframe is already defined and preprocessed
class LSTMClass():
    def __init__(self, dataframe):
        self.dataframe=dataframe
        self.results=[]

    def run(self):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.dataframe.drop('ZC=F_Open_Diff_absolute', axis=1))

        X = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))
        y = self.dataframe['ZC=F_Open_Diff_absolute'].values

        results = []  # This will store the results for the folds in the current model run
        tscv = TimeSeriesSplit(n_splits=5)

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2, recurrent_dropout=0.2),
                LSTM(100, dropout=0.2, recurrent_dropout=0.2),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

            scores = model.evaluate(X_test, y_test, verbose=0)

            results.append(scores)
        return results

    def reset_weights(self):
        K.clear_session()