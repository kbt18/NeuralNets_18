import keras
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import Dropout

import numpy as np


from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM


def evaluate_architecture(model, valdation_set):
    return None

def create_architecture(neurons, activations, input_dim, output_dim):
    if (len(neurons) != len(activations)):
        print ("Neurons must be same length as activations")
        return None

    model = Sequential()

    model.add(Dense(neurons[0], activation=activations[0], input_shape=input_dim))

    for i in range(1, len(neurons)):
        model.add(Dense(neurons[i], activation=activations[i]))

    model.add(Dense(output_dim, activation="linear"))

    model.compile(loss="mse", optimizer="adam", metrics=['mae'])

    return model

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    ############################ Question 1 ###############################
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(1024, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='linear')
    ])

    model = Sequential()

    keras.optimizers.Adam(lr=0.001)
    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
    early_stopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    np.random.shuffle(dataset)
    x, y = dataset[:, :3], dataset[:, 3:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=300, callbacks=[early_stopper])

    ############################ Question 2/3 ###############################
    neurons = [100] * 20
    activations = ["relu"] * 20

    network = create_architecture(neurons, activations, (3,), 3)

    history = network.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=300, callbacks=[early_stopper])

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
