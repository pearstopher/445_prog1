# CS445 Homework 1 Problem 11
# Christopher Juncker
# This is a sample Python script.

# "11. This is a short coding problem. You will use a perceptron with 785 inputs (including bias input)
# "and 10 outputs to learn to classify the handwritten digits in the MNIST dataset
# "(http://yann.lecun.com/exdb/mnist/). See the class slides for details of the perceptron architecture
# "and perceptron learning algorithm.
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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import exp


# "Set the learning rate to 0.1 and the momentum to 0.9.
ETA = 0.1
MOMENTUM = 0.9

# "Train your network for 50 epochs"
MAX_EPOCHS = 1


# "Experiment 1: Vary number of hidden units.
# "Do experiments with n = 20, 50, and 100.
# "(Remember to also include a bias unit with weights to every hidden and output node.)
N = 20


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
        data = pd.read_csv(dataset).to_numpy(dtype="float")
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
        return self.testing_data, self.testing_truth

    def train(self):
        return self.training_data, self.training_truth


# very simple custom confusion matrix
class ConfusionMatrix:
    def __init__(self):
        self.matrix = np.zeros((10, 10))

    def insert(self, true, pred):
        self.matrix[true][pred] += 1


# "Your neural network will have 784 inputs, one hidden layer with
# "n hidden units (where n is a parameter of your program), and 10 output units.
class NeuralNetwork:
    def __init__(self, eta):
        print("Initializing neural network...")
        # set learning rate
        self.eta = eta

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
        self.hidden_layer_weights = np.random.uniform(-0.05, 0.05, (785, N + 1))
        self.hidden_layer = np.zeros((1, N+1))
        self.hidden_layer[0] = 1  # bias
        self.output_layer_weights = np.random.uniform(-0.05, 0.05, (N+1, 10))
        self.output_layer = np.zeros((1, 10))

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

        #####################
        # FORWARD PROPAGATION
        #####################

        # for each item in the dataset
        for d, truth in zip(data[0], data[1]):

            # "For each node j in the hidden layer (i = input layer)
            # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )
            self.hidden_layer = np.dot(d, self.hidden_layer_weights)
            self.hidden_layer = np.array([self.sigmoid(x) for x in self.hidden_layer])

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            self.output_layer = np.dot(self.hidden_layer, self.output_layer_weights)
            self.output_layer = np.array([self.sigmoid(x) for x in self.output_layer])

            # (for report)
            # add our result to the confusion matrix
            if matrix:
                matrix.insert(int(d[0]), int(np.argmax(self.output_layer)))

            # "If this is the correct prediction, don’t change the weights and
            # "go on to the next training example.
            if truth == np.argmax(self.output_layer):
                num_correct += 1

            ##################
            # BACK-PROPAGATION
            ##################
            #
            # "Otherwise, update all weights in the perceptron:
            # "    𝑤i ⟵ 𝑤i + 𝜂( 𝑡(i) − 𝑦(i) ) 𝑥i(i) , where
            # "
            # "    t(i) = { 1 if the output unit is the correct one for this training example
            # "           { 0 otherwise
            # "
            # "    y(i) = { 1 if 𝒘 ∙ 𝒙(i) > 0
            # "           { 0 otherwise
            # "
            # "Thus, 𝑡(i) − 𝑦(i) can be 1, 0, or −1.
            # "
            # "(Note that this means that for some output units 𝑡(i) − 𝑦(i) could be zero,
            # " and thus the weights to that output unit would not be updated,
            # " even if the prediction was incorrect. That’s okay!)
            #
            #
            # elif not freeze:
            #     # for each perceptron
            #     for i in range(10):
            #         ti = 1 if i == d[0] else 0
            #         yi = 1 if self.outputs[i] > 0 else 0  # self.outputs[i] is already w dot x(i)
            #         # np.add(ETA*(ti - yi), self.weights) # self.weights, out=self.weights,
            #
            #         # update the weights as a function of ti, yi, and the elements in both arrays
            #         temp = d[0]
            #         d[0] = 1
            #         self.weights[i] = np.array([(wi + self.eta*(ti - yi)*xii) for wi, xii in zip(self.weights[i], d)])
            #         d[0] = temp

        # return accuracy
        return num_correct / len(data[0])

    def run(self, data, matrix, epochs):
        train_accuracy = []
        test_accuracy = []

        print("Epoch 0: ", end="")
        train_accuracy.append(self.compute_accuracy(data.train(), True))
        test_accuracy.append(self.compute_accuracy(data.train(), True))
        print("Training Set:\tAccuracy:", "{:0.5f}".format(train_accuracy[0]), end="\t")
        print("Testing Set:\tAccuracy:", "{:0.5f}".format(test_accuracy[0]))

        for i in range(epochs):
            print("Epoch " + str(i + 1) + ": ", end="")
            train_accuracy.append(self.compute_accuracy(data.train()))
            test_accuracy.append(self.compute_accuracy(data.train(), True))
            print("Training Set:\tAccuracy:", "{:0.5f}".format(train_accuracy[i + 1]), end="\t")
            print("Testing Set:\tAccuracy:", "{:0.5f}".format(test_accuracy[i + 1]))

        # "Confusion matrix on the test set, after training has been completed.
        self.compute_accuracy(data.train(), True, matrix)

        return train_accuracy, test_accuracy


def main():

    d = Data()
    p = NeuralNetwork(ETA)
    c = ConfusionMatrix()

    results = p.run(d, c, MAX_EPOCHS)

    # plot the training / testing accuracy
    plt.plot(list(range(MAX_EPOCHS + 1)), results[0])
    plt.plot(list(range(MAX_EPOCHS + 1)), results[1])
    plt.xlim([0, MAX_EPOCHS])
    plt.ylim([0, 1])
    plt.show()

    # plot the confusion matrix
    for i in range(10):
        plt.plot([-0.5, 9.5], [i+0.5, i+0.5], i, color='xkcd:chocolate', linewidth=1)  # nice colors
        plt.plot([i+0.5, i+0.5], [-0.5, 9.5], i, color='xkcd:chocolate', linewidth=1)
        for j in range(10):
            plt.scatter(i, j, s=(c.matrix[i][j] / 3), c="xkcd:fuchsia", marker="s")  # or chartreuse
            plt.annotate(int(c.matrix[i][j]), (i, j))
    plt.xlim([-0.5, 9.5])
    plt.ylim([-0.5, 9.5])
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
