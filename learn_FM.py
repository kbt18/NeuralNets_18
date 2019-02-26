import time
import keras
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import Dropout
from sklearn.model_selection import KFold

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

def create_model(neurons, activations, input_dim, output_dim):
    if (len(neurons) != len(activations)):
        print ("neurons must be same length as activations!")
        return None

    model = Sequential()

    model.add(Dense(neurons[0], activation=activations[0], input_shape=input_dim))

    for i in range(1, len(neurons)):
        model.add(Dense(neurons[i], activation=activations[i]))
        #if (i != len(neurons) - 1):
            #model.add(Dropout(0.2))

    model.add(Dense(output_dim, activation="linear"))

    model.compile(loss="mse", optimizer="adam", metrics=['mae'])

    return model

def train_and_evaluate(model, x_train, y_train, x_val, y_val, batch,
    num_epochs, learning_rate):

    keras.optimizers.Adam(lr=learning_rate)

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=20,
                                  verbose=0,
                                  restore_best_weights=True)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=batch,
              verbose=0,
              epochs=num_epochs,
              callbacks=[early_stopper])

    return model.evaluate(x_val, y_val, verbose=0)

def k_fold_cross_validation(k, x, y, model_parameters, training_parameters):
    neurons, activations, input_dim, output_dim = model_parameters
    batch_size, num_epochs, learning_rate = training_parameters

    kf = KFold(n_splits=k)

    scores = []
    i = 1
    for train_index, test_index in kf.split(x):
        print("Running Fold", i, "/", k)
        model = None

        start = time.time()
        model = create_model(neurons, activations, input_dim, output_dim)
        scores.append(train_and_evaluate(model, x[train_index], y[train_index],
                        x[test_index], y[test_index], batch_size, num_epochs,
                        learning_rate))

        end = time.time()
        print("executed in", end - start, "seconds")
        i+=1

    return np.mean(np.array(scores), axis=0)

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
    num_hidden = 2
    neurons = [1024] * num_hidden
    activations = ["relu"] * num_hidden
    model_parameters = (neurons, activations, (3,), 3)
    training_parameters = (64, 100, 0.002)
    k = 5

    mse, mae = k_fold_cross_validation(k, x, y, model_parameters, training_parameters)
    print("mean squared error:", mse)
    print("mean absolute error:", mae)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
