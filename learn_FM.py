from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

import numpy as np

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM

def train_model(x, y):



    history = model.fit(x, y, batch_size=64, epochs=1000, validation_split=0.2, callbacks=[early_stopper])


def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    model = Sequential([
        Dense(32, activation='tanh', input_shape=(3,)),
        Dense(32, activation='tanh', input_shape=(3,)),
        Dense(32, activation='tanh', input_shape=(3,)),
        Dense(3, activation='linear'),
    ])

    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
    early_stopper = EarlyStopping(patience=20, verbose=1, restore_best_weights=False)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
