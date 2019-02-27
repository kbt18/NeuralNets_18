import time
import keras
import random
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
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

def create_model(neurons=100, activation="relu", input_dim=(3,),
        output_dim=3, hidden_layers=2, learning_rate=0.001):

    model = Sequential()

    model.add(Dense(neurons, activation=activation, input_shape=input_dim))

    for i in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))
        #if (i != len(neurons) - 1):
            #model.add(Dropout(0.2))

    model.add(Dense(output_dim, activation="linear"))

    keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss="mse", optimizer="adam", metrics=['mae'])

    return model

def train_and_evaluate(model, x_train, y_train, x_val, y_val, batch,
    num_epochs):

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=20,
                                  verbose=0,
                                  restore_best_weights=True)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=batch,
              verbose=1,
              epochs=num_epochs,
              callbacks=[early_stopper])

    return model.evaluate(x_val, y_val, verbose=0)

def k_fold_cross_validation(k, x, y, model_parameters, training_parameters):
    neurons, activation, input_dim, output_dim, hidden_layers, learning_rate = model_parameters
    batch_size, num_epochs = training_parameters

    if (k <= 1): # don't cross validate if k <= 1
        split_idx = int(0.8 * len(x))

        x_train = x[:split_idx]
        y_train = y[:split_idx]
        x_val = x[split_idx:]
        y_val = y[split_idx:]

        model = create_model(neurons, activation, input_dim, output_dim, hidden_layers, learning_rate)
        mse, mae = train_and_evaluate(model, x_train, y_train, x_val, y_val,
                                        batch_size, num_epochs)
        return(mse, mae, model)

    else:
        kf = KFold(n_splits=k, shuffle=False)

        scores = []
        i = 1
        for train_index, test_index in kf.split(x):
            print("Running Fold", i, "/", k)
            model = None

            start = time.time()
            model = create_model(neurons, activation, input_dim, output_dim, hidden_layers, learning_rate)
            scores.append(train_and_evaluate(model, x[train_index], y[train_index],
                            x[test_index], y[test_index], batch_size, num_epochs,
                            learning_rate))

            end = time.time()
            print("executed in", end - start, "seconds")
            i+=1

        mse, mae = np.mean(np.array(scores), axis=0)
        return (mse, mae, model)

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    ############################ Question 1 ###############################
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(3,)),
        Dense(1024, activation='relu'),
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

    # history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100, callbacks=[early_stopper])
    #
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    ############################ Question 2/3 ###############################
    k = 1
    min_mse = 9999
    best_model = None
    best_params = None

    #random search with replacement over the following hyper-parameters
    learning_rates = (np.linspace(0.0007, 0.002, 20)).tolist()
    activations = ["relu"]
    neurons = [50, 100, 200, 400, 800, 1600]
    hidden_layers = [2, 4, 8, 16]
    epochs = [50, 200, 300, 400, 500]
    batch_sizes = [8, 16, 32, 64]

    output_layer = 3

    for i in range(200):
        learning_rate = learning_rates[random.randrange(len(learning_rates))]
        activation = activations[random.randrange(len(activations))]
        neuron = neurons[random.randrange(len(neurons))]
        hidden_layer = hidden_layers[random.randrange(len(hidden_layers))]
        epoch = epochs[random.randrange(len(epochs))]
        batch_size = batch_sizes[random.randrange(len(batch_sizes))]

        model_parameters = (neuron, activation, (3,), output_layer, hidden_layer, learning_rate)
        training_parameters = (batch_size, epoch)

        parameters = {"learning_rate":learning_rate,
                      "activation_function": activation,
                      "neurons":neuron,
                      "hidden_layers":hidden_layer,
                      "epochs":epoch,
                      "batch_size":batch_size}

        # skip models when capacity is too high
        if (neuron*hidden_layer >= 6400):
            continue

        print(parameters)
        mse, mae, model = k_fold_cross_validation(k, x, y, model_parameters, training_parameters)

        if mse < min_mse:
            min_mse = mse
            best_model = model
            best_params = parameters

    print("best mean square error", min_mse)
    print("achived with", best_params)

    best_model.save("best_model_FM.h5")

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
