import time
import keras
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
        return(train_and_evaluate(model, x_train, y_train,
                        x_val, y_val, batch_size, num_epochs))

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

        return np.mean(np.array(scores), axis=0)

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

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100, callbacks=[early_stopper])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    ############################ Question 2/3 ###############################
    k = 1

    #mse, mae = k_fold_cross_validation(k, x, y, model_parameters, training_parameters)
    #print("mean squared error:", mse)
    #print("mean absolute error:", mae)

    fig = plt.figure()
    plt.axis([0.0007, 0.002, 0, 20])
    x_axis = list()
    y_axis = list()

    learning_rates = np.linspace(0.0007, 0.002, 20)
    activation = "relu"
    neurons = 50
    hidden_layers = 2
    output_layer = 3
    for learning_rate in learning_rates:
        model_parameters = (neurons, activation, (3,), output_layer, hidden_layers, learning_rate)
        training_parameters = (64, 100)

        mse, mae = k_fold_cross_validation(k, x, y, model_parameters, training_parameters)
        x_axis.append(learning_rate)
        y_axis.append(mae)

        print("learning rate:", learning_rate)
        print("mean squared error:", mse)
        print("mean absolute error:", mae)

    plt.scatter(x_axis, y_axis)
    plt.show()


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
