# Neural Network instructions:
# input: 4 numbers (0 - 0.25), output: sum, multiplication
# example: [[.2,.2,.2,.2],[.8,.0016]]

# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform
import json
import time

import numpy as np
import spacy

from Dataset.dataset_loader import Dataset

from NeuralNetwork.Architectures.Basic import Basic
from NeuralNetwork.Architectures.LSTM.LSTM import LSTM

from NeuralNetwork.Utillities.activation_functions import Sigmoid
import random
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.Architectures.Architecture import ArchitectureType
from NeuralNetwork.Utillities.MiddleLayer import MiddleLayer
from NeuralNetwork.Utillities.ActivationLayer import ActivationLayer
from gensim import downloader

import re

EPOCHES = 1000

MIN_NUM = 0
MAX_NUM = 0.25

XOR_EXAMPLES = [[ [[0, 0]], [0]], [[[1, 0]], [1]], [[[0, 1]], [1]], [[[1, 1]], [0]]]

nlp = spacy.load("en_core_web_sm")

# def make_examples(num_of_batches, examples_per_batch): # [[[][]][][]]
#     vector = []
#     for i in range(num_of_batches):
#         vector.append([])
#         inner_vector = vector[i]
#         for j in range(examples_per_batch):
#             inner_vector.append([])
#             inner_vector2 = inner_vector[j]
#             num1 = random.uniform(MIN_NUM, MAX_NUM)
#             num3 = random.uniform(MIN_NUM, MAX_NUM)
#             num4 = random.uniform(MIN_NUM, MAX_NUM)
#             num2 = random.uniform(MIN_NUM, MAX_NUM)
#             inner_vector2.append([[num1, num2, num3, num4]])
#             inner_vector2.append([num1+num2+num3+num4, num1*num2*num3*num4])
#     return vector


def accuracy(right, current):
    return 100 - np.abs(((right - current) / right) * 100)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def machine():
    # ml = NeuralNetwork(Basic())
    # ml.add_layer(MiddleLayer(25, 16))
    # ml.add_layer(ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid))
    # ml.add_layer(MiddleLayer(16, 16))
    # ml.add_layer(ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid))
    # ml.add_layer(MiddleLayer(16, 5))
    # ml.add_layer(ActivationLayer(Sigmoid.sigmoid, Sigmoid.derivative_sigmoid))

    ml = NeuralNetwork(LSTM())

    choice = 2
    while choice == 2:
        examples = np.asarray(Dataset.make_examples(ArchitectureType.LSTM, 1, 3000))
        # print(examples.shape)
        # np.savetxt('data.csv', examples, delimiter=',')
        #
        # examples = np.loadtxt('data.csv', delimiter=',')

        count = 0

        print("Hello! 😀 I'm PsychoBot.\nMy thing is sentiment analysis.\n")
        for batch in examples:
            ml.train(batch, 3000)
            count += 1
            print('\r' + "Training 💪 - " + "{:.2f}".format(100 * (count / len(examples))) + "% | batch: " + str(
                count) + "/" + str(len(examples)), end="")
        print("\rTraining 💪 was completed successfully!")

        # # input_data = "So happy for [NAME]. So sad he's not here. Imagine this team with [NAME] instead of [NAME]. Ugh."
        # # input_data = "I believe it was a severe dislocation as opposed to a fracture. Regardless....poor guy...."
        input_data = "Wtf is this lmao god I hate reddit"
        # input_data1 = "Yes. One of her fingers is getting a sore on it and there's concern it may push her into needing braces."
        input_data2 = "Oh... I want to throw out after eating this food. I feel sick only by looking at this..."
        input_data3 = "This day was so fun! I went to the theater and it was magnificent."
        # input_data4 = "I fear from this monster. I can't sleep at night!!! I'M FREAKING OUT HELP ME"
        # input_data5 = "My cat just passed away... It's the worst day of my life. My heart is broken bro."

        check_input(input_data, ml, "1, 0, 0, 0, 0", "anger")
        # check_input(input_data1, ml, "0, 0, 1, 0, 0", "fear")
        check_input(input_data2, ml, "0, 1, 0, 0, 0", "disgust")
        check_input(input_data3, ml, "0, 0, 0, 1, 0", "joy")
        # check_input(input_data4, ml, "0, 0, 1, 0, 0", "fear")
        # check_input(input_data5, ml, "0, 0, 0, 0, 1", "sadness")

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


def check_input(input_data, ml, expectedres, expectedfeeling):
    print("\nInput: " + str(input_data))
    #
    regex = re.compile(r'[^a-zA-Z\s]')
    input_data = regex.sub('', input_data)
    input_data = input_data.lower()

    # sentence => array of words
    arr = input_data.split(" ")
    # we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    words_arr = []
    #
    model = get_model()
    #
    for word in arr:
        doc = Dataset.nlp(word)
        if doc and doc[0].is_stop:
            continue

        try:
            word_vec = model[word]
            words_arr.append(word_vec)
            # for i in range(len(word_vec)):
            #     # we_arr[i] += word_vec[i]
            #     pass
        except Exception:
            print(word + " wasn't found on word embedding.")

    res = ml.run_model(words_arr)
    highest = [0, 0]
    new_res = []
    feelings = ["anger", "disgust", "fear", "joy", "sadness"]
    for i in range(len(res)):
        if res[i] > highest[0]:
            highest[0] = res[i]
            highest[1] = i

    # for i in range(len(res)):
    #     if i == highest[1]:
    #         new_res.append(1)
    #     else:
    #         new_res.append(0)

    print("Results: " + str(res))
    print("Feeling: " + str(feelings[highest[1]]))
    # print("Wanted results: 0,0,0,0,1,0")
    # print("Wanted results: 0,0,0,0,1,0")

    # print("Wanted results: 1,0,0,0,0,0")
    # print("Wanted feeling: anger")

    print("Wanted results: ", expectedres)
    print("Wanted feeling: ", expectedfeeling)

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
            print   ("Bye bye 👋")


if __name__ == '__main__':
    main()

