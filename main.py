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
MAX_EPOCHS = 50


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
        self.training_data = self.load_set(self.TRAIN)
        self.testing_data = self.load_set(self.TEST)

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
        print("Preprocessing data...")
        # iterating one image at a time to save the true values
        for image_data in data:
            temp = image_data[0]
            # no second loop here: preprocess the whole image at once
            image_data /= max_value
            image_data[0] = temp
        return data

    def test(self):
        return self.testing_data

    def train(self):
        return self.training_data


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
        # "Choose small random initial weights, 𝑤! ∈ [−.05, .05]
        #
        # "Recall that the bias unit is always set to 1,
        # "and the bias weight is treated like any other weight.
        self.hidden_layer_weights = np.array([np.random.uniform(-0.05, 0.05, 785) for _ in range(N + 1)])
        self.hidden_layer = np.random.uniform(-0.05, 0.05, N + 1)  # hidden units
        self.hidden_layer[0] = 1  # hidden bias unit value
        self.output_layer_weights = np.array(np.random.uniform(-0.05, 0.05, N) for _ in range(10 + 1))
        self.output_layer = np.zeros(10)
        self.eta = eta

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

        # "For each node j in the hidden layer (i = input layer)
        # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )

        # "For each node k in the output layer (j = hidden layer)
        # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )

        # for each item in the dataset
        for d in data:

            # "For each node j in the hidden layer (i = input layer)
            # h_j = σ ( Σ_i ( w_ji x_i + w_j0 ) )
            for j in range(N):
                # save the true value and replace with input bias
                temp = d[0]
                d[0] = 1

                # Compute 𝒘 ∙ 𝒙 (i) at each hidden unit.
                # self.hidden_layer[j + 1] = np.dot(self.hidden_layer_weights[j], d)  # +1 to skip bias unit

                total = 0
                for i in range(d):
                    total += np.dot(self.hidden_layer_weights[j + 1][i], d[i]) + self.hidden_layer_weights[0]

                d[0] = temp

                # apply the sigmoid activation function to the result
                # self.hidden_layer[j + 1] = self.sigmoid(self.hidden_layer[j + 1])
                self.hidden_layer[j + 1] = self.sigmoid(total)

            # "For each node k in the output layer (j = hidden layer)
            # o_k = σ ( Σ_j ( w_kj h_j + w_k0 ) )
            for k in range(10):
                # the bias value is already built in to this array
                # Compute 𝒘 ∙ 𝒙 (i) at each output unit.
                # self.output_layer[k] = np.dot(self.output_layer_weights[k], self.hidden_layer)

                total = 0
                for j in range(self.hidden_layer):
                    total += np.dot(self.output_layer_weights[k + 1][j]) + self.output_layer_weights[0]

                self.output_layer[k + 1] = self.sigmoid(total)

            # (for report)
            # add our result to the confusion matrix
            if matrix:
                matrix.insert(int(d[0]), int(np.argmax(self.output_layer)))

            # "If this is the correct prediction, don’t change the weights and
            # "go on to the next training example.
            if d[0] == np.argmax(self.output_layer):
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
        return num_correct / len(data.data)  # data.data or data?

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
