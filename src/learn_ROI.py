import pickle
import h5py


import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
#from sklearn.model_selection import GridSearchCV



# from nn_lib import (
#     MultiLayerNetwork,
#     Trainer,
#     Preprocessor,
#     save_network,
#     load_network,
# )

from illustrate import illustrate_results_ROI
from tensorflow.python.estimator import keras


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
def model_params(num_hidden, num_neurons_inlayer, activation, final_activation, learning_rate): #, epochs, batch_size):
    model = Sequential([
        Dense(num_neurons_inlayer, activation=activation, input_shape=(3,)), # 3 input angles
    ])
    # add hidden layers, ma
    for i in range(num_hidden):
        model.add(Dense(num_neurons_inlayer, activation=activation))
    # # add last layer
    model.add(Dense(4, activation=final_activation)) # output is 4 because there are 4 regions
    keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc']) # binary_crossentropy,

    #train model
    # early_stopper = EarlyStopping(patience=20, verbose=0, restore_best_weights=False)
    # history = model.fit(x, y, batch_size=data.shape[0], epochs=epochs, validation_split=0.2, callbacks=[early_stopper], verbose=0)
    return model

def model_train(model, data, x, y, epochs=1000):
    early_stopper = EarlyStopping(patience=20, verbose=0, restore_best_weights=False)
    history = model.fit(x, y, batch_size=len(data)//100, epochs=epochs, validation_split=0.2, callbacks=[early_stopper], verbose=0)

# ------------------------------------------------------------------------------------------------------
def train_baseline(dataset, prep):
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

    neurons = 1
    activations = "sigmoid"
    hiddenlayers = 1
    final_activation = "softmax"
    model = model_params(hiddenlayers, neurons, activations, final_activation)
    model.summary()
    model_train(model, data, x, y, 1000)
    eval_result = model.evaluate(x_val, y_val) # return [loss, metrics]
    results.append((eval_result, activations, hiddenlayers, neurons))
    print(eval_result)
    return model

# ------------------------------------------------------------------------------------------------------
def evaluate_architecture(dataset, prep):
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

    # 300 times
    # 1. random search ------------------------------------------------------------------------------------------------------
    final_activations_ = ["softmax"]
    activations_ = ["selu"]
    hiddenlayers_ = np.arange(0, 16)
    neurons_ = np.arange(1, 264)
    learning_rate_ = np.arange(0.0001, 0.04, 40)

    # 100 times
    # 1. random search ------------------------------------------------------------------------------------------------------
    # final_activations_ = ["linear", "softmax"]
    # activations_ = ["tanh", "relu", "sigmoid", "selu"]
    # hiddenlayers_ = np.arange(0, 16)
    # neurons_ = np.arange(1, 500)
    # learning_rate_ = np.arange(0.0001, 0.04, 40)

    # 30 times each to make graph
    # 1. hiddenlayers_ ------------------------------------------------------------------------------------------------------
    # final_activations_ = ["linear"]
    # activations_ = ["relu"]
    # hiddenlayers_ = np.arange(0, 16)
    # neurons_ = [30]
    # learning_rate_ = [0.01]

    # # 2. neurons_------------------------------------------------------------------------------------------------------
    # final_activations_ = ["linear"]
    # activations_ = ["relu"]
    # hiddenlayers_ = [2]
    # neurons_ = np.arange(1, 256)
    # learning_rate_ = [0.01]

    # 3. learning_rate_------------------------------------------------------------------------------------------------------
    # final_activations_ = ["linear"]
    # activations_ = ["relu"]
    # hiddenlayers_ = [2]
    # neurons_ = [30]
    # learning_rate_ = np.arange(0.00005, 0.05, 50)


    # epochs_ = np.arange(100, 1000)
    # batch_size_ = np.arange(2, 64)

    # epochs 100 - 500 # batch size 2 - 64 # dropout perhaps - only good if overfitting
    # if num of neurons * layers = cap > 10 000 then skip
    # network capacity - if its too high it leads to overfitting

    for n in range(300):
        i_final_activation = np.random.randint(0, len(final_activations_))
        i_activation = np.random.randint(0, len(activations_))
        i_hiddenlayer = np.random.randint(0, len(hiddenlayers_))
        i_neurons = np.random.randint(0, len(neurons_))
        i_learning_rate = np.random.randint(0, len(learning_rate_))
        # i_epochs = np.arange(0, len(epochs_))
        # i_batch_size = np.arange(0, len(batch_size_))

        model = model_params(hiddenlayers_[i_hiddenlayer], neurons_[i_neurons], activations_[i_activation], final_activations_[i_final_activation], learning_rate_[i_learning_rate]) #, batch_size_[i_batch_size])
        model_train(model, data, x, y, 1000)
        eval_result = model.evaluate(x_val, y_val)
        results.append((eval_result, activations_[i_activation], hiddenlayers_[i_hiddenlayer], neurons_[i_neurons], final_activations_[i_final_activation], learning_rate_[i_learning_rate])) #, batch_size_[i_batch_size]))


    with open("./model_valid.bin", "wb") as f:
        pickle.dump(results, f)

    results2 = None
    with open("./model_valid.bin", "rb") as f:
        results2 = pickle.load(f)
    for tuple in results2:
        print(tuple)

    model.save("./my_model_1.300.h5")
    return model

def predict_hidden(dataset):
    model = model_params(hiddenlayers=2, neurons=7, activations="sigmoid", final_activation="softmax")
    model.load("./roi_model.dat")
    return model.predict(dataset)

# ------------------------------------------------------------------------------------------------------
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

    neurons = 2
    activations = "relu"
    hiddenlayers = 7
    final_activation = "softmax"
    model = model_params(hiddenlayers, neurons, activations, final_activation)
    model.summary()
    model_train(model, data, x, y, 1000)
    eval_result = model.evaluate(x_val, y_val) # return [loss, metrics]
    results.append((eval_result, activations, hiddenlayers, neurons))
    print(eval_result)
    return model

def batch_size_max():
    dataset = np.loadtxt("ROI_dataset.dat")
    n = len(dataset)
    r1 = np.sum(dataset[:,3])
    r2 = np.sum(dataset[:,4])
    r3 = np.sum(dataset[:,5])
    r4 = np.sum(dataset[:,6])
    P1 = r1/n # 0.0
    P2 = r2/n
    P3 = r3/n # should be represented in each minibatch at least once
    P4 = r4/n
    print(P1, P2, P3, P4) # 0.0848 0.097408 0.009216 0.808576 - very skewed
    print(1/P3)
    min_batch_size = len(dataset)*P3 # max batch size = len(dataset) ~= 150k
    print(min_batch_size) # = batch_size needs to be len(dataset)//min_batch_size

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

    model = evaluate_architecture(dataset, prep)

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
    # batch_size_max()
    main()
