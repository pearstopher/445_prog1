# CS445 Programming Assignment #1
# Christopher Juncker
#
# "For this homework you will implement a two-layer neural network (i.e, one hidden-layer) to
# "perform the handwritten digit recognition task of Homework 1. Please write your own neural
# "network code; don’t use code written by others, though you can refer to other code if you
# "need help understanding the algorithm. You may use whatever programming language you prefer.
#
#
# MNIST data in CSV format:
# https://pjreddie.com/projects/mnist-in-csv/
#
# Data files in data/ folder:
#   mnist_train.csv
#   mnist_test.csv
# (Not included in commit to save space)
#
#

import os
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from math import exp

# "Set the learning rate to 0.1
ETA = 0.1
# "Train your network for 50 epochs"
MAX_EPOCHS = 50


# class for loading and preprocessing MNIST data
# data is contained in a numpy array
class Data:
    def __init__(self):
        self.TRAIN = "data/mnist_train.csv"
        self.TEST = "data/mnist_test.csv"
        self.training_data, self.training_truth = self.load_set(self.TRAIN)
        self.testing_data, self.testing_truth = self.load_set(self.TEST)

    def load_set(self, dataset):
        print("Reading '" + dataset + "' data set...")
        data = read_csv(dataset).to_numpy(dtype="float")
        return self.preprocess(data)

    # "Preprocessing: Scale each data value to be between 0 and 1.
    # "(i.e., divide each value by 255, which is the maximum value in the original data)
    # "This will help keep the weights from getting too large.
    @staticmethod
    def preprocess(data):
        max_value = 255
        ground_truth = np.empty(len(data))
        print("Preprocessing data...")
        # iterating one image at a time
        for i in range(len(data)):
            # save the true value
            ground_truth[i] = data[i][0]
            # set the bias
            data[i][0] = max_value  # (this will end up as 1)

        # now it is safe to normalize ALL the image data at once
        data /= max_value
        return data, ground_truth

    def test(self):
        # randomly permute the test data
        indices = np.arange(self.testing_data.shape[0])
        np.random.shuffle(indices)
        self.testing_data = self.testing_data[indices]
        self.testing_truth = self.testing_truth[indices]

        return self.testing_data, self.testing_truth

    def train(self, limit=0):
        # randomly permute the training data
        indices = np.arange(self.training_data.shape[0])
        np.random.shuffle(indices)
        self.training_data = self.training_data[indices]
        self.training_truth = self.training_truth[indices]

        if limit:
            return self.training_data[0:limit], self.training_truth[0:limit]
        return self.training_data, self.training_truth


# very simple custom confusion matrix
class ConfusionMatrix:
    def __init__(self):
        self.matrix = np.zeros((10, 10))

    def insert(self, true, predicted):
        self.matrix[true][predicted] += 1


# "Your neural network will have 784 inputs, one hidden layer with
# "n hidden units (where n is a parameter of your program), and 10 output units.
class NeuralNetwork:
    def __init__(self, hidden_units, momentum, training_examples):
        print("Initializing neural network...")
        # set learning rate and momentum
        self.eta = ETA
        self.hidden_units = hidden_units
        self.momentum = momentum
        self.training_examples = training_examples

        # explicitly set the size of these arrays (for matrix multiplication / my own sanity)
        # input layer:                              1 x 785
        # hidden layer weights:                     785 x (N + 1)
        # input layer (dot) hidden layer weights:   1 x (N + 1)
        #
        # hidden layer:                             1 x (N + 1)
        # output layer weights:                     (N + 1) x 10
        # hidden layer (dot) output layer weights:  1 x 10
        #
        #
        # "Choose small random initial weights, 𝑤! ∈ [−.05, .05]
        self.hidden_layer_weights = np.random.uniform(-0.05, 0.05, (785, self.hidden_units + 1))
        self.hidden_layer_weights_change = np.zeros((785, self.hidden_units + 1))  # save momentum
        self.hidden_layer = np.zeros((self.hidden_units+1))
        self.hidden_layer[0] = 1  # bias
        self.output_layer_weights = np.random.uniform(-0.05, 0.05, (self.hidden_units + 1, 10))
        self.output_layer_weights_change = np.zeros((self.hidden_units + 1, 10))  # save momentum
        self.output_layer = np.zeros(10)

    # "The activation function for each hidden and output unit is the sigmoid function
    # σ(z) = 1 / ( 1 + e^(-z) )
    @staticmethod
    def sigmoid(value):
        activation = 1 / (1 + exp(-value))
        return activation

    # "Compute the accuracy on the training and test sets for this initial set of weights,
    # "to include in your plot. (Call this “epoch 0”.)
    def compute_accuracy(self, data, freeze=False, matrix=None):
        num_correct = 0

        # for each item in the dataset
        for d, truth in zip(data[0], data[1]):

            #####################
            # FORWARD PROPAGATION
            #####################

            # "For each node j in the hidden layer (i = input layer)
            # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )
            self.hidden_layer = np.dot(d, self.hidden_layer_weights)
            self.hidden_layer = np.array([self.sigmoid(x) for x in self.hidden_layer])
            # print(self.hidden_layer)  # these are the right size as expected

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            self.output_layer = np.dot(self.hidden_layer, self.output_layer_weights)
            self.output_layer = np.array([self.sigmoid(x) for x in self.output_layer])
            # print(self.output_layer)  # these are the right size as expected

            # (for report)
            # add our result to the confusion matrix
            if matrix:
                matrix.insert(int(np.argmax(self.output_layer)), int(truth), )  # x=predicted, y=true

            # "If this is the correct prediction, don’t change the weights and
            # "go on to the next training example.
            if truth == np.argmax(self.output_layer):
                num_correct += 1

            ##################
            # BACK-PROPAGATION
            ##################

            elif not freeze:
                output_error = np.empty(10)
                hidden_error = np.empty(self.hidden_units + 1)

                # "For each output unit k, calculate error term δ_k
                # δ_k <- o_k (1 - o_k) (t_k - o_k)
                #
                # t = true value
                for k, o_k in enumerate(self.output_layer):
                    # "Set the target value tk for output unit k to 0.9 if the input class is the kth class,
                    # "0.1 otherwise
                    t_k = 0.9 if truth == k else 0.1  # had truth == o_k instead of truth == the index

                    error = o_k * (1 - o_k) * (t_k - o_k)
                    output_error[k] = error

                # "for each hidden unit j, calculate error term δ_j (k = output units)
                # δ_j <- h_j (1 - h_j) (Σ_k ( w_kj * δ_k ) )
                for j, h_j in enumerate(self.hidden_layer):
                    hidden_error[j] = np.dot(self.output_layer_weights[j], output_error)
                    hidden_error[j] *= h_j * (1 - h_j)

                # "Hidden to Output layer: For each weight w_kj
                # w_kj = w_kj + Δw_kj
                # Δw_kj = η * δ_k * h_j
                self.output_layer_weights_change = \
                    self.eta * (self.hidden_layer.reshape(self.hidden_units + 1, 1) @ output_error.reshape(1, 10)) + \
                    self.momentum * self.output_layer_weights_change
                self.output_layer_weights += self.output_layer_weights_change

                # "Input to Hidden layer: For each weight w_ji
                # w_ji = w_ji + Δw_ji
                # Δw_ji = η * δ_j * x_i
                self.hidden_layer_weights_change = \
                    self.eta * (d.reshape(785, 1) @ hidden_error.reshape(1, self.hidden_units + 1)) + \
                    self.momentum * self.hidden_layer_weights_change
                self.hidden_layer_weights += self.hidden_layer_weights_change

        # return accuracy
        return num_correct / len(data[0])

    def run(self, data, matrix, epochs):
        train_accuracy = []
        test_accuracy = []

        print("Epoch 0: ", end="")
        train_accuracy.append(self.compute_accuracy(data.train(self.training_examples), True))
        test_accuracy.append(self.compute_accuracy(data.test(), True))
        print("Training Set:\tAccuracy:", "{:0.5f}".format(train_accuracy[0]), end="\t")
        print("Testing Set:\tAccuracy:", "{:0.5f}".format(test_accuracy[0]))

        for i in range(epochs):
            print("Epoch " + str(i + 1) + ": ", end="")
            train_accuracy.append(self.compute_accuracy(data.train(self.training_examples)))
            test_accuracy.append(self.compute_accuracy(data.test(), True))
            print("Training Set:\tAccuracy:", "{:0.5f}".format(train_accuracy[i + 1]), end="\t")
            print("Testing Set:\tAccuracy:", "{:0.5f}".format(test_accuracy[i + 1]))

        # "Confusion matrix on the test set, after training has been completed.
        self.compute_accuracy(data.test(), True, matrix)

        return train_accuracy, test_accuracy


def main():
    # load the data (just once)
    d = Data()

    # run all of the exercises in a row
    # (hidden units, momentum, training examples)
    trials = [(100, 0.9,  60000),
              (50,  0.9,  60000),
              (25,  0.9,  60000),
              (100, 0.5,  60000),
              (100, 0.25, 60000),
              (100, 0,    60000),
              (100, 0.9,  30000),
              (100, 0.9,  15000),
              ]

    for w, (x, y, z) in enumerate(trials):
        p = NeuralNetwork(x, y, z)
        c = ConfusionMatrix()

        results = p.run(d, c, MAX_EPOCHS)

        # make sure the /images/ directory exists
        d = "images"
        exist = os.path.exists(d)
        if not exist:
            os.makedirs(d)

        # plot the training / testing accuracy
        plt.figure(figsize=(10, 10))
        plt.plot(list(range(MAX_EPOCHS + 1)), results[0])
        plt.plot(list(range(MAX_EPOCHS + 1)), results[1])
        plt.xlim([0, MAX_EPOCHS])
        plt.ylim([0, 1])
        plt.savefig("images/exercise" + str(w+1) + "_" + str(x) + "_" + str(y) + "_" + str(z) + ".png")

        # plot the confusion matrix
        plt.clf()
        for i in range(10):
            plt.plot([-0.5, 9.5], [i+0.5, i+0.5], i, color='xkcd:chocolate', linewidth=1)  # nice colors
            plt.plot([i+0.5, i+0.5], [-0.5, 9.5], i, color='xkcd:chocolate', linewidth=1)
            for j in range(10):
                plt.scatter(i, j, s=(c.matrix[i][j] * 2.7), c="xkcd:fuchsia", marker="s")  # or chartreuse
                plt.annotate(int(c.matrix[i][j]), (i, j))
        plt.xlim([-0.5, 9.5])
        plt.ylim([-0.5, 9.5])
        plt.gca().invert_yaxis()
        plt.savefig("images/exercise" + str(w+1) + "_" + str(x) + "_" + str(y) + "_" + str(z) + "_cm.png")


if __name__ == '__main__':
    main()
