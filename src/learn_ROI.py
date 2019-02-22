import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
#from sklearn.model_selection import GridSearchCV



from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from illustrate import illustrate_results_ROI

def network_test(np_array):
    assert(np_array.shape[1] == 3)
    return np.random.rand(np_array.shape[0], 4)

# apply min max scaling norm
class Preproc:
    def __init__(self, x):
        pass
    def apply(self, x):
        return x

    def revert(self, x):
        return x

# create an architecture for the model
def model_params(data, x, y, num_hidden, num_neurons_inlayer, activation, final_activation):
    model = Sequential([
        Dense(num_neurons_inlayer, activation=activation, input_shape=(3,)), # 3 input angles
    ])

    # add hidden layers
    for i in range(num_hidden -1):
        model.add(Dense(num_neurons_inlayer, activation=activation))
    # # add last layer
    model.add(Dense(4, activation=final_activation)) # output is 4 because there are 4 regions

    #train model
    model.compile(loss="mse", optimizer="adam", metrics=['mae'])
    early_stopper = EarlyStopping(patience=20, verbose=1, restore_best_weights=False)
    history = model.fit(x, y, batch_size=data.shape[0], epochs=1000, validation_split=0.2, callbacks=[early_stopper], verbose=0)
    return model

def train_model(dataset, prep):
    splitindex = int(dataset.shape[0] * 0.2)
    test = dataset[:splitindex, :]
    train_dataset = dataset[splitindex:,:]
    # k-fold split - from cw1
    x = train_dataset[:,0:3]
    y = train_dataset[:,3:7]
    x_val = test[:,0:3]
    y_val = test[:,3:7]

    data = prep.apply(dataset)
    results = []
    # iterate through different architectures

    # neurons = 1
    # activations = "selu"
    # hiddenlayers = 1
    # model = model_params(data, x, y, hiddenlayers, neurons, activations, final_activation)
    # model.summary()
    # eval_result = model.evaluate(x_val, y_val) # return [loss, metrics]
    # results.append((eval_result, activations, hiddenlayers, neurons))

    for final_activation in ["linear","softmax"]:
        for activations in ["tanh", "relu", "sigmoid", "hard_sigmoid", "selu"]:
            for hiddenlayers in range(1,12):
                for neurons in range(1, 10):
                    model=model_params(data, x, y, hiddenlayers, neurons, activations, final_activation)
                    eval_result = model.evaluate(x_val, y_val)
                    results.append((eval_result, activations, hiddenlayers, neurons, final_activation))

    for tuple in results:
        # i+=1
        print(tuple)
        # print(results[i])

    model.save("./roi_model.dat")
    return model

def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    # aim: predict which zone the pointer arm is based on the angular position of the arm segments
    # arm segment angles: columns 0,1,2, one hot encoded labels: 3,4,5,6
    #######################################################################

    prep = Preproc(dataset)
    # mlnet = MultiLayerNetwork(prep)
    # save_network(network)

    model = train_model(dataset, prep)

    def network(three_angle):
        return model.predict(three_angle)
    # np.random.shuffle(dataset)
    # x, y = dataset[:, :3], dataset[:, 3:]
    # model = train_model(x, y)
    # model.save("./fm_model.dat")

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # graph model - 10 random locations - prediction vs reality check
    # illustrate_results_ROI(network, prep)


if __name__ == "__main__":
    main()
