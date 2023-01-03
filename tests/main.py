# Help:
# matrix multiplication - np.dot
# initialize array - np.zeros
# initialize random values array - np.random.uniform
import json
import time
import math
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

BATCHES = 1
EXAMPLES = 2000

XOR_EXAMPLES = [[[[0, 0]], [0]], [[[1, 0]], [1]], [[[0, 1]], [1]], [[[1, 1]], [0]]]

nlp = spacy.load("en_core_web_sm")

amount_true_feel = [0, 0, 0, 0, 0]
amount_false_feel = [0, 0, 0, 0, 0]
amount_false_feel_inv = [0, 0, 0, 0, 0]


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


def machine(answer, list_of_feelings):
    ml = NeuralNetwork(LSTM(list_of_feelings))

    while True:
        if answer:
            dataset_path, examples = Dataset.save_dataset(ArchitectureType.LSTM, BATCHES, EXAMPLES, 'data.npy',
                                                          list_of_feelings)
            with open('list.json', 'w') as f:
                json.dump(list_of_feelings, f)
        examples = np.load('data.npy', allow_pickle=True)

        count = 0

        print("Hello! 😀 I'm PsychoBot.\nMy thing is sentiment analysis.\n")
        for batch in examples:
            ml.train(batch, math.floor(0.9 * EXAMPLES))
            count += 1
            print('\r' + "Training 💪 - " + "{:.2f}".format(100 * (count / len(examples))) + "% | batch: " + str(
                count) + "/" + str(len(examples)), end="")
        print("\rTraining 💪 was completed successfully!")
        amount_true = 0

        # ["anger", "disgust", "fear", "joy", "sadness"]
        # anger, disgust, fear, joy, sadness

        batchlen = 0
        for batch in examples:
            batchlen = len(batch) - math.floor(0.9 * EXAMPLES)
            for example in batch[math.floor(0.9 * EXAMPLES):]:
                ls = []
                up_index = 0
                for i in range(len(list_of_feelings)):
                    ls.append(example[1][i])
                    if i > 0:
                        if ls[i] > ls[i - 1]:
                            up_index = i

                if check_input(example[0], ml, str(ls), list_of_feelings[up_index], up_index):
                    amount_true += 1

        # print(amount_true, EXAMPLES)

        for i in range(len(list_of_feelings)):
            try:
                print("Success for " + list_of_feelings[i] + ": " + str(
                    (100 * amount_true_feel[i]) / (amount_true_feel[i] + amount_false_feel[i])) + "%")
            except Exception as e:
                print("Success for " + list_of_feelings[i] + ": There was no such feeling")

            try:
                print("Inverse success for " + list_of_feelings[i] + ": " + str(
                    (100 * amount_true_feel[i]) / (amount_true_feel[i] + amount_false_feel_inv[i])) + "%")
            except Exception as e:
                print("Inverse success for " + list_of_feelings[i] + ": Didn't even guess ;)")
            print()

        print("General percents of success: " + str((100 * amount_true) / batchlen) + "%")
        print()

        print(
            "Are the results fulfilling your satisfaction?\n1 - Yes. The student became the master\n2 - No. Learn more!")
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


def check_input(input_data, ml, expectedres, expectedfeeling, expectedfeeling_num, list_of_feelings):
    return_val = False
    # print("\nInput: " + str(input_data))
    #
    # regex = re.compile(r'[^a-zA-Z\s]')
    # input_data = regex.sub('', input_data)
    # input_data = input_data.lower()

    # sentence => array of words
    # arr = input_data.split(" ")
    # we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # words_arr = []
    #
    # model = get_model()
    #
    # for word in arr:
    #     # doc = Dataset.nlp(word)
    #     if doc and doc[0].is_stop:
    #         continue
    #
    #     try:
    #         word = doc[0].lemma_
    #         word_vec = model[word]
    #         words_arr.append(word_vec)
    #         # for i in range(len(word_vec)):
    #         #     # we_arr[i] += word_vec[i]
    #         #     pass
    #     except Exception:
    #         print(word + " wasn't found on word embedding.")

    res = ml.run_model(input_data)
    highest = [0, 0]
    new_res = []

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
    print("Feeling: " + str(list_of_feelings[highest[1]]))

    if str(list_of_feelings[highest[1]]) == str(expectedfeeling):
        amount_true_feel[highest[1]] += 1
        return_val = True
    else:
        amount_false_feel_inv[highest[1]] += 1
        amount_false_feel[expectedfeeling_num] += 1

    # print("Wanted results: 0,0,0,0,1,0")
    # print("Wanted results: 0,0,0,0,1,0")

    # print("Wanted results: 1,0,0,0,0,0")
    # print("Wanted feeling: anger")

    print("Wanted results: ", expectedres)
    print("Wanted feeling: ", expectedfeeling)
    print()
    return return_val


def get_model():
    # use pre-trained model and use it
    model = downloader.load('glove-twitter-25')

    return model


def main():
    print('\033[1m' + "PsychoBot POC 1.3" + '\033[0m' + "\nAll rights reserved © PsychoBot 2022\n")
    choice = 0
    ml = None
    while choice != 3:
        print(
            "1 - Train the machine on prepared dataset\n2 - Train the machine on un-prepared dataset\n3 - Use working machine\n4 - I've seen enough")
        choice = int(input())
        if choice == 1:
            if ml is not None:
                print(
                    "Training a new machine will overwrite the previous machine you've made.\nAre you sure you want to train a new one? Y/n")
                choice = input()
                print("")
                if choice.lower() == "y":
                    with open('list_of_feelings.json', 'r') as f:
                        list_of_feelings = json.load(f)
                    ml = machine(False, list_of_feelings)
            else:
                with open('list_of_feelings.json', 'r') as f:
                    list_of_feelings = json.load(f)
                ml = machine(False, list_of_feelings)
        elif choice == 2:
            print("Enter the list of feelings:")
            list_of_feelings = input()
            list_of_feelings = list(list_of_feelings.split(", "))
            print(list_of_feelings)
            ml = machine(True, list_of_feelings)
        elif choice == 3:
            print("This is not possible yet")
        elif choice == 4:
            print("Bye bye 👋")


if __name__ == '__main__':
    main()
