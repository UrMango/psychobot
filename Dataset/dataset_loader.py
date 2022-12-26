import numpy as np
import pandas as pd
from gensim import models, similarities, downloader
import logging
import re
import spacy
from NeuralNetwork.Architectures.Architecture import ArchitectureType


class Dataset:
    df = pd.read_csv(r'E:\GitHub\hadera-801-psychobot\Dataset\go_emotions_dataset.csv')
    last_count = 0
    nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def make_examples(architecture, num_of_batches, examples_per_batch):
        """
		Makes examples by a specified number of batches and examples per batch
		Number:param num_of_batches:
		Number:param examples_per_batch:
		:return:
		"""
        model = Dataset.get_model()

        anger = 0
        disgust = 0
        fear = 0
        joy = 0
        sadness = 0
        neutral = 0
        vector = []

        try:
            for i in range(num_of_batches):
                print("BATCH:", i, "/", num_of_batches)
                vector.append([])
                inner_vector = vector[i]
                for j in range(examples_per_batch):
                    inner_vector2 = []

                    while (
                            Dataset.df.example_very_unclear[Dataset.last_count] == 'TRUE'
                            or (Dataset.df.anger[Dataset.last_count] == 0
                                and Dataset.df.disgust[Dataset.last_count] == 0
                                and Dataset.df.fear[Dataset.last_count] == 0
                                and Dataset.df.joy[Dataset.last_count] == 0
                                and Dataset.df.sadness[Dataset.last_count] == 0)):
                        # and Dataset.df.neutral[Dataset.last_count] == 0):
                        # print(Dataset.df.anger[Dataset.last_count], Dataset.df.disgust[Dataset.last_count],
                              # Dataset.df.fear[Dataset.last_count], Dataset.df.joy[Dataset.last_count],
                              # Dataset.df.neutral[Dataset.last_count])
                        Dataset.last_count += 1
                    # sentence
                    text = Dataset.df.text[Dataset.last_count]
                    # print(Dataset.df.anger[Dataset.last_count])
                    # print(Dataset.df.disgust[Dataset.last_count])
                    # print(Dataset.df.fear[Dataset.last_count])
                    # print(Dataset.df.joy[Dataset.last_count])
                    # print(Dataset.df.sadness[Dataset.last_count])

                    if Dataset.df.anger[Dataset.last_count] != 0:
                        anger += 1
                    if Dataset.df.disgust[Dataset.last_count] != 0:
                        disgust += 1
                    if Dataset.df.fear[Dataset.last_count] != 0:
                        fear += 1
                    if Dataset.df.joy[Dataset.last_count] != 0:
                        joy += 1
                    if Dataset.df.sadness[Dataset.last_count] != 0:
                        sadness += 1
                    # if Dataset.df.neutral[Dataset.last_count] != 0:
                    # neutral += 1

                    regex = re.compile(r'[^a-zA-Z\s]')
                    text = regex.sub('', text)
                    text = text.lower()

                    # sentence => array of words
                    arr = text.split(" ")

                    we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    words_arr = []
                    res = []

                    # words => word embedding
                    for word in arr:
                        doc = Dataset.nlp(word)
                        if doc and doc[0].is_stop:
                            continue
                        try:
                            # print("Before:", word)
                            word = doc[0].lemma_
                            # print("After:", word)
                            word_vec = model[word]
                            if architecture == ArchitectureType.LSTM:
                                words_arr.append(word_vec)
                                continue
                            for k in range(len(word_vec)):
                                if architecture == ArchitectureType.BASIC:
                                    we_arr[k] += word_vec[k]
                        except Exception:
                            pass
                        # print(word + " wasn't found on word embedding.")

                    if architecture == ArchitectureType.BASIC:
                        for k in range(len(we_arr)):
                            we_arr[k] /= 25

                    # 28 (6) feelings vector res
                    res.append(Dataset.df.anger[Dataset.last_count])
                    res.append(Dataset.df.disgust[Dataset.last_count])
                    res.append(Dataset.df.fear[Dataset.last_count])
                    res.append(Dataset.df.joy[Dataset.last_count])
                    res.append(Dataset.df.sadness[Dataset.last_count])
                    # res.append(Dataset.df.neutral[Dataset.last_count])

                    print('\r' + str(j) + "/" + str(examples_per_batch), end="")

                    arr_final = []
                    if architecture == ArchitectureType.BASIC:
                        if len(we_arr) == 0:
                            Dataset.last_count += 1
                            continue
                        inner_vector2.append(we_arr)
                    elif architecture == ArchitectureType.LSTM:
                        if len(words_arr) <= 1:
                            Dataset.last_count += 1
                            continue
                        inner_vector2.append(words_arr)
                    inner_vector2.append(res)

                    inner_vector.append(inner_vector2.copy())

                    Dataset.last_count += 1
        except Exception:
            pass

        print(anger, disgust, fear, joy, sadness)
        return vector

    @staticmethod
    def get_model():

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # use pre-trained model and use it
        model = downloader.load('glove-twitter-25')

        return model

    @staticmethod
    def save_dataset(arch_type, batches, examples, file_name):
        examples = Dataset.make_examples(arch_type, batches, examples)

        np.save(file_name, examples)

        return file_name
