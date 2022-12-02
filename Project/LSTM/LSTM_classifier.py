import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Masking, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from sklearn.metrics import accuracy_score

from tools.tools import evaluate as evalate


tf.get_logger().setLevel('INFO')
random.seed(2022)
np.random.seed(2022)
tf.random.set_seed(2022)

class LSTMclf:
    def __init__(self, input_shape, lstm_size=200, lr=1e-3): 
        self.input_shape = input_shape
        self.lstm_size = lstm_size
        self.lr = lr
        self.trained = False
        self.optimizer = Adam(learning_rate=self.lr)
        self.define_model()

    def define_model(self):
        self.model = Sequential([
            Input(shape=self.input_shape[1:]),
            Masking(mask_value=-10),
            LSTM(self.lstm_size),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', 
                           metrics=['accuracy'])

    def trainCV(self, X, y, num_folds, batch_size, epochs=10, verbose=True):
        subset_size = len(X) // num_folds
        val_labels = []
        pred_labels = []
        if verbose:
            self.define_model()
            message = f'Beginning training LSTM with {num_folds}-fold validation...'
            print(message)
        for i in range(num_folds):
            trainMatrix = X[:i*subset_size] + X[(i+1)*subset_size:]
            valMatrix = X[i*subset_size:(i+1)*subset_size]

            trainLabels = y[:i*subset_size]  + y[(i+1)*subset_size:]
            valLabels = y[i*subset_size:(i+1)*subset_size]
            val_labels.extend(valLabels)

            self.model.fit(trainMatrix, 
                           trainLabels,
                           epochs=epochs, 
                           batch_size=batch_size,
                           verbose=int(verbose))
            
            pred = self.model.predict(valMatrix).reshape(-1)
            pred = (pred > 0.5).astype(int)
            pred_labels.extend(pred)
            
            if verbose:
                train_acc = self.model.evaluate(trainMatrix, trainLabels)[1]
                val_acc = self.model.evaluate(valMatrix, valLabels)[1]
                print(f'Fold {i+1} Train: {train_acc * 100: .2f}% Val: {val_acc * 100: .2f}%')

        metrics = evalate(val_labels, pred_labels)
        self.trained = True
        return metrics

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        self.model.fit(X_train,
                       y_train, 
                       batch_size=batch_size, 
                       epochs=epochs,
                       verbose=1)
        self.trained = True

    def evaluate(self, X, y, batch_size):
        assert self.trained, 'Model has not been trained yet.'
        pred = self.model.predict(X).reshape(-1)
        pred = (pred > 0.5).astype(int)
        metrics = eval(y, pred)
        return pred, metrics


class BiLSTMclf(LSTMclf):
    def define_model(self):
        self.model = Sequential([
            Input(shape=self.input_shape[1:]),
            Masking(mask_value=-10),
            Bidirectional(LSTM(self.lstm_size)),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', 
                           metrics=['accuracy'])
