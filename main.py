# Neural Network instructions:
# input: 4 numbers (0 - 1), output: sum, multiplication
# example: [[.1,.2,.3,.4],[1.0,2.4]]

# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform

import numpy as np
from activation_functions import Tanh
import random
from NeuralNetwork import NeuralNetwork
from MiddleLayer import MiddleLayer
from ActivationLayer import ActivationLayer

EPOCHES = 1000

MIN_NUM = 0
MAX_NUM = 0.25

XOR_EXAMPLES = [[ [[0, 0]], [0]], [[[1, 0]], [1]], [[[0, 1]], [1]], [[[1, 1]], [0]]]

def make_examples(num_of_batches, examples_per_batch): # [[[][]][][]]
    vector = []
    for i in range(num_of_batches):
        vector.append([])
        inner_vector = vector[i]
        for j in range(examples_per_batch):
            inner_vector.append([])
            inner_vector2 = inner_vector[j]
            num1 = random.uniform(MIN_NUM, MAX_NUM)
            num3 = random.uniform(MIN_NUM, MAX_NUM)
            num4 = random.uniform(MIN_NUM, MAX_NUM)
            num2 = random.uniform(MIN_NUM, MAX_NUM)
            inner_vector2.append([[num1, num2, num3, num4]])
            inner_vector2.append([num1+num2+num3+num4, num1*num2*num3*num4])
    return vector


def main():
    print('\033[1m' + "PsychoBot POC 1.0" + '\033[0m' + "\nAll rights reserved © PsychoBot 2022\n")
    # print("1 - Train the machine\n2 - Use an existing model")
    # ml = None
    # choice = int(input())
    # if choice == 1:
    #     ml = NeuralNetwork()
    #     examples = make_examples(20, 100)
    #     print("Starting learning...")
    #     print(ml.learning(examples))
    #     print("Machine learned: " + str(len(examples) * len(examples[0])) + " Examples")
    # else:
    #     pass #vect = input("")
    #
    # print("Write 4 numbers")
    # num1 = float(input())
    # num2 = float(input())
    # num3 = float(input())
    # num4 = float(input())
    # output = ml.activate([num1, num2, num3, num4])
    # print("THE RESULTS:\nSum: " + str(output[0]) + "\nMultiplication: " + str(output[1]) + "\n")

    ml = NeuralNetwork()
    ml.add_layer(MiddleLayer(4, 5))
    ml.add_layer(ActivationLayer(Tanh.tanh, Tanh.tanh_derivative))
    ml.add_layer(MiddleLayer(5, 5))
    ml.add_layer(ActivationLayer(Tanh.tanh, Tanh.tanh_derivative))
    ml.add_layer(MiddleLayer(5, 2))
    ml.add_layer(ActivationLayer(Tanh.tanh, Tanh.tanh_derivative))
    choice = 2
    while choice == 2:
        examples = make_examples(20, 1000)
        for batch in examples:
            ml.train(batch)

        print("\nInput: 0.1, 0.2, 0.1, 0.2")
        print("Results: " + str(ml.run_model([0.1, 0.2, 0.1, 0.2])))
        print("Wanted results: " + str([[0.1 + 0.2 + 0.1 + 0.2], [0.1*0.2*0.1*0.2]]))
        print("\n")

        print("Are the results fulfilling your satisfaction?\n1 - Yes. The student became the master\n2 - No. Learn more!")
        choice = int(input())
        if choice == 1:
            print("HURRAY!\nA NEW MASTER HAS ARRIVED...")
            print("""
   ___                _                              _            
  / _ \\___ _   _  ___| |__   ___     /\\/\\   __ _ ___| |_ ___ _ __ 
 / /_)/ __| | | |/ __| '_ \\ / _ \\   /    \\ / _` / __| __/ _ \\ '__|
/ ___/\\__ \\ |_| | (__| | | | (_) | / /\\/\\ \\ (_| \\__ \\ ||  __/ |   
\\/    |___/\__, |\\___|_| |_|\\___/  \\/    \\/\\__,_|___/\\__\\___|_|   
           |___/                                                  
            """)
        if choice == 2:
            print("i'm sorry... I'll learn more /:")


if __name__ == '__main__':
    main()

