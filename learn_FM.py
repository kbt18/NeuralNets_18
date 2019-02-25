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

def create_architecture(neurons, activations, input_dim):
    if (len(neurons) != len(activations)):
        print ("Neurons must be same length as activations")
        return None

    model = Sequential()

    model.add(neurons[0], activation=activations[0], input_shape=input_dim)

    for i in range(1, len(neurons)):
        model.add(neurons[i], activation=activations[i])

    return model

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    ############################ Question 1 ###############################
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(2048, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='linear')
    ])

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

    history = model.fit(x, y, validation_split=0.2, batch_size=32, epochs=300, callbacks=[early_stopper])
    #history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=100, callbacks=[early_stopper])
    #print(model.evaluate(x_val, y_val))

    ############################ Question 2/3 ###############################


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
