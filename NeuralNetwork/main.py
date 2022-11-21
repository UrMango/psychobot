# Neural Network instructions:
# input: 4 numbers (0 - 0.25), output: sum, multiplication
# example: [[.2,.2,.2,.2],[.8,.0016]]

# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform
import time

import numpy as np

from Dataset.dataset_loader import Dataset
from activation_functions import Tanh
import random
from NeuralNetwork import NeuralNetwork
from MiddleLayer import MiddleLayer
from ActivationLayer import ActivationLayer
from gensim import downloader

import re

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


def accuracy(right, current):
    return 100 - np.abs(((right - current) / right) * 100)


def machine():
    ml = NeuralNetwork()
    ml.add_layer(MiddleLayer(25, 16))
    ml.add_layer(ActivationLayer(Tanh.tanh, Tanh.tanh_derivative))
    ml.add_layer(MiddleLayer(16, 16))
    ml.add_layer(ActivationLayer(Tanh.tanh, Tanh.tanh_derivative))
    ml.add_layer(MiddleLayer(16, 28))
    ml.add_layer(ActivationLayer(Tanh.tanh, Tanh.tanh_derivative))
    choice = 2
    while choice == 2:
        examples = Dataset.make_examples(3000, 100)
        count = 0

        print("Hello! 😀 I'm PsychoBot POC.\nMy current expectations are to find sum and multiplications of 4 numbers between 0 to 0.25.\n")
        for batch in examples:
            ml.train(batch)
            count += 1
            print('\r' + "Training 💪 - " + "{:.2f}".format(100 * (count / len(examples))) + "% | batch: " + str(
                count) + "/" + str(len(examples)), end="")
        print("\rTraining 💪 was completed successfully!")

        input_data = "wanted to downvote this, but it's not fault homie."
        print("\nInput: " + str(input_data))

        regex = re.compile(r'[^a-zA-Z\s]')
        input_data = regex.sub('', input_data)
        input_data = input_data.lower()

        # sentence => array of words
        arr = input_data.split(" ")
        we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        model = get_model()

        for word in arr:
            if (
                    word == "was"
                    or word == "is"
                    or word == "am"
                    or word == "i"
                    or word == "are"
                    or word == "we"
                    or word == "them"
                    or word == "were"
                    or word == "they"
                    or word == "your"
            ):
                continue

            try:
                word_vec = model[word]
                for i in range(len(word_vec)):
                    we_arr[i] += word_vec[i]
            except Exception:
                print(word + " wasn't found on word embedding.")

        res = ml.run_model([we_arr])
        print("Results: " + str(res))
        print("Wanted results: 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")

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
            time.sleep(2)
            return ml
        if choice == 2:
            print("i'm sorry... I'll learn more /:\n")


def get_model():
    # use pre-trained model and use it
    model = downloader.load('glove-twitter-25')

    return model


def main():
    print('\033[1m' + "PsychoBot POC 1.3" + '\033[0m' + "\nAll rights reserved © PsychoBot 2022\n")
    choice = 0
    ml = None
    while choice != 3:
        print("1 - Train the machine\n2 - Use working machine\n3 - I've seen enough")
        choice = int(input())
        if choice == 1:
            if ml is not None:
                print("Training a new machine will overwrite the previous machine you've made.\nAre you sure you want to train a new one? Y/n")
                choice = input()
                print("")
                if choice == "Y" or choice == "y":
                    ml = machine()
            else:
                ml = machine()
        elif choice == 2:
            if ml is None:
                print("There's no trained machine :(")
            else:
                print("\nTesting the trained machine\n\nWrite 4 numbers between 0 - 0.25:")
                num1 = float(input())
                num2 = float(input())
                num3 = float(input())
                num4 = float(input())

                input_data = [num1, num2, num3, num4]
                print("\nInput: " + str(input_data))
                res = ml.run_model(input_data)
                print("Results: " + str(res))
                wanted_res = [[input_data[0] + input_data[1] + input_data[2] + input_data[3]],
                              [input_data[0] * input_data[1] * input_data[2] * input_data[3]]]
                print("Wanted results: " + str(wanted_res))
                print("Accuracy: 1st - " + "{:.2f}".format(
                    accuracy(wanted_res[0][0], res[0][0])) + "% | 2nd - " + "{:.2f}".format(
                    accuracy(wanted_res[1][0], res[0][1])) + "%\n")
        elif choice == 3:
            print("Bye bye 👋")


if __name__ == '__main__':
    main()

