import time
import keras
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import compute_class_weight

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM
from keras import backend as K

def predict_hidden(dataset):
    model = load_model('best_model_ROI.h5')
    y_pred = model.predict(dataset)
    y_zeros = np.zeros_like(y_pred)
    y_zeros[np.arange(len(y_zeros)), y_pred.argmax(1)] = 1
    return(y_zeros)

def create_model(neurons=100, activation="relu", input_dim=(3,),
        output_dim=4, hidden_layers=2, learning_rate=0.001):

    model = Sequential()

    model.add(Dense(neurons, activation=activation, input_shape=input_dim))

    for i in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))

    model.add(Dense(output_dim, activation="softmax"))

    keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['acc'])

    return model


def _compute_class_weight_dictionary(y):
    # helper for returning a dictionary instead of an array
    classes = np.unique(y)
    class_weight = compute_class_weight("balanced", classes, y)
    class_weight_dict = dict(zip(classes, class_weight))
    return class_weight_dict



def train_and_evaluate(model, x_train, y_train, x_val, y_val, batch,
                       num_epochs):

    early_stopper = EarlyStopping(monitor='val_loss',
                                  patience=20,
                                  verbose=0,
                                  restore_best_weights=True)


    y_val_vec = [np.where(r == 1)[0][0] for r in y_val]


    cw_dict = _compute_class_weight_dictionary(y_val_vec)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size=batch,
              verbose=1,
              epochs=num_epochs,
              callbacks=[early_stopper],
              class_weight=cw_dict)


    y_pred = model.predict(x_val)
    y_pred_vec = np.zeros_like(y_pred)
    y_pred_vec[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

    y_pred_vec = [np.where(r == 1)[0][0] for r in y_pred_vec]

    cm = confusion_matrix(y_val_vec, y_pred_vec, sample_weight=None)
    f1 = f1_score(y_val_vec, y_pred_vec, average='macro')

    cce, acc = model.evaluate(x_val, y_val, verbose=0)

    return (cce, acc, f1, cm)


def k_fold_cross_validation(k, x, y, model_parameters, training_parameters):
    neurons, activation, input_dim, output_dim, hidden_layers, learning_rate = model_parameters
    batch_size, num_epochs = training_parameters

    if (k <= 1): # don't cross validate if k <= 1
        split_idx = int(0.8 * len(x))

        x_train = x[:split_idx]
        y_train = y[:split_idx]
        x_val = x[split_idx:]
        y_val = y[split_idx:]

        # sm = SMOTE(random_state=2)
        # x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

        model = create_model(neurons, activation, input_dim, output_dim, hidden_layers, learning_rate)
        cce, acc, f1, cm = train_and_evaluate(model, x_train, y_train, x_val, y_val, batch_size, num_epochs)
        return(cce, acc, f1, cm, model)

    else:
        kf = KFold(n_splits=k, shuffle=False)

        scores = []
        i = 1
        for train_index, test_index in kf.split(x):
            print("Running Fold", i, "/", k)
            model = None

            start = time.time()
            model = create_model(neurons, activation, input_dim, output_dim, hidden_layers, learning_rate)
            cce, acc, f1, cm = (train_and_evaluate(model, x[train_index], y[train_index],
                            x[test_index], y[test_index], batch_size, num_epochs,
                            learning_rate))

            scores.append([cce, acc, f1])

            end = time.time()
            print("executed in", end - start, "seconds")
            i+=1

        cce, acc = np.mean(np.array(scores), axis=0)
        return (cce, acc, f1, cm, model)

def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    ############################ Question 1 ###############################
    # model = Sequential([
    #     Dense(1024, activation='relu', input_shape=(3,)),
    #     Dense(1024, activation='relu'),
    #     Dense(3, activation='linear')
    # ])
    #
    # keras.optimizers.Adam(lr=0.001)
    # keras.optimizers.RMSprop(lr=0.001)
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mae'])
    # early_stopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    np.random.shuffle(dataset)
    x, y = dataset[:, :3], dataset[:, 3:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=100, epochs=100, callbacks=[early_stopper])
    #
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    ############################ Question 2/3  ###############################
    ############################ PLOTS #######################################

    # # see how learning rate affects accuracy
    # lr_history = []
    # mse_history = []
    #
    # neuron= 128
    # hidden_layer = 4
    # k = 1
    # learning_rates = (np.linspace(0.00005, 0.05, 50)).tolist()
    # for lr in learning_rates:
    #     model_parameters = (neuron, "relu", (3,), 3, hidden_layer, lr)
    #     training_parameters = (100, 100)
    #
    #     mse, mae, model = k_fold_cross_validation(1, x, y, model_parameters, training_parameters)
    #
    #     lr_history.append(lr)
    #     mse_history.append(mse)
    #
    # plt.scatter(lr_history, mse_history)
    # plt.show()

    # # see how neurons affects accuracy
    # neuron_history = []
    # mse_history = []
    # neurons = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # lr = 0.001
    # for neuron in neurons:
    #     model_parameters = (neuron, "relu", (3,), 3, 4, lr)
    #     training_parameters = (100, 100)
    #
    #     mse, mae, model = k_fold_cross_validation(1, x, y, model_parameters, training_parameters)
    #
    #     neuron_history.append(neuron)
    #     mse_history.append(mse)
    #
    # plt.scatter(neuron_history, mse_history)
    # plt.show()

    # # see how hidden_layers affects accuracy
    # layer_history = []
    # mse_history = []
    # hidden_layers = range(1, 17, 1)
    # lr = 0.001
    # for hidden_layer in hidden_layers:
    #     model_parameters = (128, "relu", (3,), 3, hidden_layer, lr)
    #     training_parameters = (100, 100)
    #
    #     mse, mae, model = k_fold_cross_validation(1, x, y, model_parameters, training_parameters)
    #
    #     layer_history.append(hidden_layer)
    #     mse_history.append(mse)
    #
    # plt.scatter(layer_history, mse_history)
    # plt.show()

    # see how batch_sizes affects accuracy
    #########################################################################################
    # batch_history = []
    # mse_history = []
    # hidden_layer = 4
    # batch_sizes = [2, 4, 8, 16, 32, 64, 128, 512, 1024]
    # lr = 0.001
    # for batch in batch_sizes:
    #     model_parameters = (128, "relu", (3,), 3, hidden_layer, lr)
    #     training_parameters = (batch, 100)
    #
    #     mse, mae, model = k_fold_cross_validation(1, x, y, model_parameters, training_parameters)
    #
    #     batch_history.append(batch)
    #     mse_history.append(mse)
    #
    # plt.scatter(batch_history, mse_history)
    # plt.show()


    k = 1
    f1_max = -1
    best_model = None
    best_params = None

    # #random search with replacement over the following hyper-parameters
    # learning_rates = (np.linspace(0.00005, 0.05, 50)).tolist()
    # activations = ["relu"]
    # neurons = [50, 100, 200, 400]
    # hidden_layers = [2, 4]
    # epochs = [100]
    # batch_sizes = [128]

    ###########################################################################################
    learning_rates = (np.linspace(0.00005, 0.05, 50)).tolist()
    final_activations = ["softmax"]
    activations = ["relu"]
    hidden_layers = np.arange(1, 8)
    neurons = np.arange(1, 64)
    epochs = [100]
    batch_sizes = np.arange(64, 128)
    ###########################################################################################

    output_layer = 4
    results = []

    out = open("randsearch_roi_res.txt", "w")

    for i in range(70):
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

        # print(parameters)
        cce, acc, f1, cm, model = k_fold_cross_validation(k, x, y, model_parameters, training_parameters)

        eval_result = [cce, acc]
        results.append((eval_result, activation, hidden_layer, neuron, "softmax", f1))

        # out.write(str(parameters)+ " "+str(cce) + " " + str(acc) +" "+  str(f1) + " "+ str(cm)+"\n" )
        out.write("("+str(eval_result)+", \'"+str(activation)+"\', "+str(hidden_layer)+", "+str(neuron)+", \'"+str('softmax')+"\', "+str(f1)+")\n")
        # print(cm)

        if f1 > f1_max:
            f1_max = f1
            best_model = model
            best_params = parameters
            best_conf_matrix = cm

    out.close()
    print("best f1", f1_max)
    print("achived with", best_params)
    print("best cm", best_conf_matrix)

    best_model.save("best_model_ROI.h5")
    print (results)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    #illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
