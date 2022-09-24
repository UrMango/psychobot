# Neural Network instructions:
# input: 4 numbers (0 - 1), output: sum, multiplication
# example: [[.1,.2,.3,.4],[1.0,2.4]]

# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform

import numpy as np
import random

SIZE_MIDDLE_FIRST_LAYER = 5
NUMBER_OF_OUTPUTS = 2
NUMBER_OF_INPUTS = 4

def activation(x):
    return x


def activation_array(array):
    res = []
    for item in array:
        res.append(activation(item))
    return res


def sigmoid(x):
    return 1 / (1+np.exp(-1*x))


def inverse_sigmoid(x):
    return np.log(x/1-x)


def derivative_sigmoid(x):
    return np.exp(-x) / pow((1+np.exp(-x)), 2)


class NeuralNetwork:
    # Fields
    input = np.zeros(NUMBER_OF_INPUTS)
    weights_input = np.zeros((SIZE_MIDDLE_FIRST_LAYER, NUMBER_OF_INPUTS))
    biases_input = np.zeros(SIZE_MIDDLE_FIRST_LAYER)

    neurons_first_layer = np.zeros(SIZE_MIDDLE_FIRST_LAYER)
    weights_first_layer = np.zeros((NUMBER_OF_OUTPUTS, SIZE_MIDDLE_FIRST_LAYER))
    biases_first_layer = np.zeros(NUMBER_OF_OUTPUTS)
    output = np.zeros(NUMBER_OF_OUTPUTS)

    # Constructor
    def __init__(self, vector=None):
        if vector:
            self.weights_input = vector[0]
            self.weights_first_layer = vector[1]
            self.biases_input = vector[2]
            self.biases_first_layer = vector[3]
        else:
            self.weights_input = np.random.uniform(low=-1, high=1, size=(SIZE_MIDDLE_FIRST_LAYER, NUMBER_OF_INPUTS))
            self.weights_first_layer = np.random.uniform(low=-1, high=1, size=(NUMBER_OF_OUTPUTS, SIZE_MIDDLE_FIRST_LAYER))
            self.biases_input = np.random.uniform(low=-5, high=5, size=SIZE_MIDDLE_FIRST_LAYER)
            self.biases_first_layer = np.random.uniform(low=-5, high=5, size=NUMBER_OF_OUTPUTS)

    def activate(self, inputs):
        self.input = inputs
        self.neurons_first_layer = activation_array(np.add(np.dot(self.weights_input, self.input), self.biases_input))
        self.output = activation_array(np.add(np.dot(self.weights_first_layer, self.neurons_first_layer), self.biases_first_layer))
        return self.output

    def cost(self, examples_vect):
        return_value = 0

        for example in examples_vect:
            for i in range(NUMBER_OF_OUTPUTS):
                return_value += pow((example[1][i] - self.activate(example[0])[i]), 2)
        return_value /= len(examples_vect)
        return return_value

    def nudge(self, vect_nudges):
        # ([[2,3][1,2]], [3,4] , [[7,6][1,5]], [7,8])
        weights_input_nudge = np.zeros((SIZE_MIDDLE_FIRST_LAYER, NUMBER_OF_INPUTS))
        biases_input_nudge = np.zeros(SIZE_MIDDLE_FIRST_LAYER)
        weights_first_layer_nudge = np.zeros((NUMBER_OF_OUTPUTS, SIZE_MIDDLE_FIRST_LAYER))
        biases_first_layer_nudge = np.zeros(NUMBER_OF_OUTPUTS)
        for nudge in vect_nudges:
            weights_input_nudge = np.add(nudge[0], weights_input_nudge)
            biases_input_nudge = np.add(nudge[1], biases_input_nudge)
            weights_first_layer_nudge = np.add(nudge[2], weights_first_layer_nudge)
            biases_first_layer_nudge = np.add(nudge[3], biases_first_layer_nudge)

        factor = -1 / len(vect_nudges)

        weights_input_nudge = np.dot(weights_input_nudge, factor)
        biases_input_nudge = np.dot(biases_input_nudge, factor)
        weights_first_layer_nudge = np.dot(weights_first_layer_nudge, factor)
        biases_first_layer_nudge = np.dot(biases_first_layer_nudge, factor)

        self.weights_input = np.add(weights_input_nudge, self.weights_input)
        self.biases_input = np.add(biases_input_nudge, self.biases_input)
        self.weights_first_layer = np.add(weights_first_layer_nudge, self.weights_first_layer)
        self.biases_first_layer = np.add(biases_first_layer_nudge, self.biases_first_layer)

    def learning(self, examples):
        return_vect = []
        nudges = []
        for batch in examples:
            for example in batch:
                print("$")
                nudges.append(self.backpropagation(example))
            self.nudge(nudges)
            # print(self.cost(examples[-1]))
            nudges = []
        return_vect.append(self.weights_input)
        return_vect.append(self.biases_input)
        return_vect.append(self.weights_first_layer)
        return_vect.append(self.biases_first_layer)
        return return_vect

    def backpropagation(self, example):
        """
        Function for changing the biases and weights for the next activation
        :return: Vector with 2 matrices and 2 vectors
        """
        output = self.activate(example[0])
        weights_first_layer_nudge = np.zeros((NUMBER_OF_OUTPUTS, SIZE_MIDDLE_FIRST_LAYER))
        for i in range(NUMBER_OF_OUTPUTS):
            for j in range(SIZE_MIDDLE_FIRST_LAYER):
                weights_first_layer_nudge[i][j] = self.neurons_first_layer[j]*2*(output[i]-example[1][i]) #* derivative_sigmoid(inverse_sigmoid(output[i]))

        biases_first_layer_nudge = np.zeros(NUMBER_OF_OUTPUTS)
        for i in range(NUMBER_OF_OUTPUTS):
            biases_first_layer_nudge[i] = 1 * 2 * (output[i] - example[1][i]) #* derivative_sigmoid(inverse_sigmoid(output[i]))

        first_layer_nudge = np.zeros(SIZE_MIDDLE_FIRST_LAYER)
        for i in range(SIZE_MIDDLE_FIRST_LAYER):
            for j in range(NUMBER_OF_OUTPUTS):
                first_layer_nudge[i] += self.weights_first_layer[j][i] * 2 * (output[j] - example[1][j]) #* derivative_sigmoid(inverse_sigmoid(output[j]))

        weights_input_nudge = np.zeros((SIZE_MIDDLE_FIRST_LAYER, NUMBER_OF_INPUTS))
        for i in range(SIZE_MIDDLE_FIRST_LAYER):
            for j in range(NUMBER_OF_INPUTS):
                weights_input_nudge[i][j] = self.input[j] * first_layer_nudge[i] #* derivative_sigmoid(inverse_sigmoid(self.neurons_first_layer[i]))

        biases_input_nudge = np.zeros(SIZE_MIDDLE_FIRST_LAYER)
        for i in range(SIZE_MIDDLE_FIRST_LAYER):
            biases_input_nudge[i] = 1 * first_layer_nudge[i] * derivative_sigmoid(inverse_sigmoid(self.neurons_first_layer[i]))
        print("$")
        return [weights_input_nudge, biases_input_nudge, weights_first_layer_nudge, biases_first_layer_nudge]


def make_examples(num_of_batches, examples_per_batch): # [[[][]][][]]
    vector = []
    for i in range(num_of_batches):
        vector.append([])
        inner_vector = vector[i]
        for j in range(examples_per_batch):
            inner_vector.append([])
            inner_vector2 = inner_vector[j]
            num1 = random.uniform(0, 1)
            num3 = random.uniform(0, 1)
            num4 = random.uniform(0, 1)
            num2 = random.uniform(0, 1)
            inner_vector2.append([num1, num2, num3, num4])
            inner_vector2.append([num1+num2+num3+num4, num1*num2*num3*num4])
    return vector


def main():
    print('\033[1m' + "PsychoBot POC 1.0" + '\033[0m' + "\nAll rights reserved © PsychoBot 2022\n")
    print("1 - Train the machine\n2 - Use an existing model")
    choice = int(input())
    if choice == 1:
        ml = NeuralNetwork()
        examples = make_examples(20, 1000)
        print("Starting learning...")
        print(ml.learning(examples))
        print("Machine learned: " + str(len(examples) * len(examples[0])) + " Examples")
    else:
        pass #vect = input("")

    print("Write 4 numbers")
    num1 = float(input())
    num2 = float(input())
    num3 = float(input())
    num4 = float(input())
    output = ml.activate([num1, num2, num3, num4])
    print("THE RESULTS:\nSum: " + str(output[0]) + "\nMultiplication: " + str(output[1]) + "\n")
    print("Are the results fulfilling your satisfaction?\n1 - Yes. The student became the master\n2 - No. Learn more!")
    choice = int(input())


if __name__ == '__main__':
    main()

