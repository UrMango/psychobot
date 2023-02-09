import numpy as np
import pandas as pd
from gensim import models, similarities, downloader
import logging
import re
import random
import spacy
from NeuralNetwork.Architectures.Architecture import ArchitectureType

EXACT_PATH = r'hadera-801-psychobot\Dataset\final_dataset.csv'


class Dataset:
    path = r'E:\GitHub'
    #path = r'C:\Users\magshimim\Documents\Magshimim\Psychobot'

    df = pd.read_csv(path + "\\" + EXACT_PATH)

    last_count = 0
    nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def make_examples(architecture, num_of_examples, list_of_feelings):
        """
		Makes examples by a specified number of batches and examples per batch
		Number:param num_of_examples:
		List:param list_of_feelings: list of emotions names to learn
		:return:
		"""
        model = Dataset.get_model()

        emotions_count = []
        for i in range(len(list_of_feelings)):
            emotions_count.append(0)
        # anger = 0
        # disgust = 0
        # fear = 0
        # joy = 0
        # sadness = 0
        # neutral = 0
        vector = []

        try:
            for j in range(num_of_examples):
                inner_vector = []
                stop_grow = False

                for k, emotion2 in enumerate(list_of_feelings):
                    # Check if the value of the current emotion column at the last count index is 1
                    # and if the current emotion count is greater than (num_of_examples / 5)
                    if (getattr(Dataset.df, emotion2)[Dataset.last_count] == 1 and emotions_count[k] >= (
                            num_of_examples / len(list_of_feelings))):
                        stop_grow = True
                        break

                while (Dataset.df.example_very_unclear[Dataset.last_count] == 'TRUE' or
                       all(getattr(Dataset.df, emotion)[Dataset.last_count] == 0 for emotion in
                           list_of_feelings)) or stop_grow:
                    # (Dataset.df.id[Dataset.last_count] != "e2718281-mango-god" and Dataset.df.id[Dataset.last_count] != "pi31415-42-69")
                    Dataset.last_count += 1
                    stop_grow = False
                    for k, emotion2 in enumerate(list_of_feelings):
                        # Check if the value of the current emotion column at the last count index is 1
                        # and if the current emotion count is greater than (num_of_examples / 5)
                        if (getattr(Dataset.df, emotion2)[Dataset.last_count] == 1 and emotions_count[k] >= (
                                num_of_examples / len(list_of_feelings))):
                            stop_grow = True
                            break

                # sentence
                text = Dataset.df.text[Dataset.last_count]

                for k, emotion in enumerate(list_of_feelings):
                    if getattr(Dataset.df, emotion)[Dataset.last_count] != 0:
                        emotions_count[k] += 1

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

                if architecture == ArchitectureType.BASIC:
                    for k in range(len(we_arr)):
                        we_arr[k] /= 25

                # 28 (6) feelings vector res
                for k, emotion in enumerate(list_of_feelings):
                    res.append(getattr(Dataset.df, emotion)[Dataset.last_count])

                print('\r' + "Creating dataset 💪 - " + "{:.2f}".format(100 * j / num_of_examples) + "% | examples: " + str(j) + "/" + str(num_of_examples), end="")

                arr_final = []
                if architecture == ArchitectureType.BASIC:
                    if len(we_arr) == 0:
                        Dataset.last_count += 1
                        continue
                    inner_vector.append(we_arr)
                elif architecture == ArchitectureType.LSTM:
                    if len(words_arr) <= 1:
                        Dataset.last_count += 1
                        continue
                    inner_vector.append(words_arr)
                inner_vector.append(res)

                vector.append(inner_vector.copy())

                Dataset.last_count += 1

                count_sum = 0
                for emotion in emotions_count:
                    count_sum += emotion

                if count_sum == num_of_examples:
                    break
        except Exception as e:
            print("\nexception")
            print(Dataset.last_count)
            print(e)

        random.shuffle(vector)
        print()

        print(emotions_count)
        return vector

    @staticmethod
    def shuffle_dataset(data_location, data_name):
        df = pd.read_csv(data_location + '\\' + data_name, header=0)  # r'E:\GitHub\hadera-801-psychobot\Dataset\
        shuffled_df = df.sample(frac=1)
        shuffled_df.to_csv('final_dataset.csv', index=False)

    @staticmethod
    def merge_dataset(data_location, data_name):
        new_df = pd.read_csv(data_location + '\\' + data_name) # r'E:\GitHub\hadera-801-psychobot\Dataset\

        dict_feelings = {"id": [], "text": [], "example_very_unclear": [], "admiration": [], "amusement": [],
                         "anger": [], "annoyance": [], "approval": [], "caring": [], "confusion": [], "curiosity": [],
                         "desire": [], "disappointment": [], "disapproval": [], "disgust": [], "embarrassment": [],
                         "enthusiasm": [], "fear": [], "gratitude": [], "grief": [], "happy": [], "love": [],
                         "worry": [], "optimism": [], "pride": [], "realization": [], "relief": [], "remorse": [],
                         "sadness": [], "neutral": []}

        # feelings

        # admiration
        # amusement
        # anger
        # annoyance
        # approval
        # caring
        # confusion
        # curiosity
        # desire
        # disappointment
        # disapproval
        # disgust
        # embarrassment
        # excitement - enthusiasm
        # fear
        # gratitude
        # grief
        # joy - happiness - fun - happy
        # love
        # nervousness - worry
        # optimism
        # pride
        # realization
        # relief
        # remorse
        # sadness
        # surprise
        # neutral neutral

        for i in range(len(new_df.Text)):
            # if new_df.sentiment[i] in ["empty", "boredom", "hate"]:
            #     continue
            dict_feelings["id"].append("e2718281-mango-god")
            dict_feelings["example_very_unclear"].append("FALSE")
            dict_feelings["text"].append(new_df.Text[i])

            print('\r' + "Merging 💪 - " + "{:.2f}".format(100 * (i / len(new_df.Text))) + "% | numeric: " + str(
                i) + "/" + str(len(new_df.Text)), end="")
            for emotion in dict_feelings.keys():
                if emotion not in ["id", "example_very_unclear", "text"]:
                    if new_df.Emotion[i] == emotion:
                        dict_feelings[emotion].append(1)
                    else:
                        dict_feelings[emotion].append(0)

            # if new_df.sentiment[i] == "fun":
            #     dict_feelings["happiness"][-1] = 1

        df_to_append = pd.DataFrame(dict_feelings)

        df_to_append.to_csv(data_location + r'\go_emotions_dataset-remade-remade.csv', mode='a', index=False, header=False)
        print('\r' + "Merging 💪 Completed successfully")

    @staticmethod
    def get_model():

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # use pre-trained model and use it
        model = downloader.load('glove-twitter-25')

        return model

    @staticmethod
    def save_dataset(arch_type, examples, file_name, list_of_feelings):
        examples = Dataset.make_examples(arch_type, examples, list_of_feelings)

        np.save(file_name, examples)

        return file_name, examples


if __name__ == '__main__':
    Dataset.shuffle_dataset(r'E:\GitHub\hadera-801-psychobot\Dataset', 'go_emotions_dataset-remade-remade.csv')
