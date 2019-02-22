from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping


# evaluate architecture run lots of times for different paramters and the paramter_split might split the data differently each time
# also we want to return how well the model is performing

def evaluate_architecture(x_train, y_train, x_val, y_val, layers=[32, 32, 32],
                          activation_func='tanh', optimizer='adam', loss='mse'):
    model = Sequential()
    # adding the first layer should be outside of the for loop because it needs to take the input_shape, unlike the other layers
    model.add(Dense(layers[0], activation=activation_func, input_shape=(3,)))
    for neurons in layers[1:]:
        model.add(Dense(neurons, activation=activation_func))
    model.add(Dense(3, activation='linear'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])
    early_stopper = EarlyStopping(patience=20, verbose=1, restore_best_weights=False)
    history = model.fit(x_train, y_train, batch_size=64, epochs=1000,
                        callbacks=[early_stopper], verbose=1, validation_data=(x_val, y_val))
    # could plot these graphs here, which shows how the loss and validation loss decreases over the epochs
    # epochs is on the x axis and the loss and val_loss are on the y axis
    #     plt.plot(history.history['loss'])
    #     plt.plot(history.history['val_loss'])
    return model.evaluate(x_val, y_val, batch_size=256), model

def train_model(x, y):
    model = Sequential([
        Dense(32, activation='tanh', input_shape=(3,)),
        Dense(32, activation='tanh', input_shape=(3,)),
        Dense(32, activation='tanh', input_shape=(3,)),
        # output later ahas a linear activiation function because relu and sigmoid don't allow more than 1 or less than 0
        Dense(3, activation='linear'),
    ])

    # model.summary()

    model.compile(loss="mse", optimizer="adam", metrics=['mae'])

    # train the model, batch size is how many examples we look at before updating the weights,
    # epochs is how often we want to iterate over our whole dataset
    # EarlyStopping stops the training if the validation loss doesn't improve anymore so it doesn't actually get to 1000
    # patience specificies how many epochs we wait to get better validaiton loss before we stop training
    early_stopper = EarlyStopping(patience=20, verbose=1, restore_best_weights=False)
    history = model.fit(x, y, batch_size=64, epochs=1000, validation_split=0.2, callbacks=[early_stopper])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    #
    # plt.plot(history.history['mean_absolute_error'])
    # plt.plot(history.history['val_mean_absolute_error'])
    # make predictions based on the input data, we are just using our data x here but we should be using the data they give us
    return model

def main():
    dataset = np.loadtxt("FM_dataset.dat")

     # Question 1

    np.random.shuffle(dataset)
    x, y = dataset[:, :3], dataset[:, 3:]
    model = train_model(x, y)
    model.save("./fm_model.dat")
    # make prediction using model using input z
    # model.predict(z)

    # illustrate_results_FM(network, prep)


    # Question 2

    # Need to split split training dataset and validation set manually because the keras
    # shuffle only shuffles after splitting off the group in the validation set

    # nb_train_examples = int(len(x) * 0.8)
    # x_train, y_train = x[:nb_train_examples], y[:nb_train_examples]
    # x_val, y_val = x[nb_train_examples:], y[nb_train_examples:]
    #
    # # evaluate_architecture returns them model so we can save it and compare with others
    # # to find best parameters
    # metrics, model_e = evaluate_architecture(x_train, y_train, x_val, y_val,
    #                                        layers=[32, 16], activation_func='tanh',
    #                                        loss='mae')
    # print(metrics)

if __name__ == "__main__":
    main()



# import numpy as np
#
# from nn_lib import (
#     MultiLayerNetwork,
#     Trainer,
#     Preprocessor,
#     save_network,
#     load_network,
# )
# from illustrate import illustrate_results_FM
#
#
# def main():
#     dataset = np.loadtxt("FM_dataset.dat")
#     #######################################################################
#     #                       ** START OF YOUR CODE **
#     #######################################################################
#
#     #######################################################################
#     #                       ** END OF YOUR CODE **
#     #######################################################################
#     illustrate_results_FM(network, prep)
#
#
# if __name__ == "__main__":
#     main()
