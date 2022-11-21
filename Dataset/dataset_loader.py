import pandas as pd
from gensim import models, similarities, downloader
import logging
import re

class Dataset:
	df = pd.read_csv(r'E:\GitHub\hadera-801-psychobot\Dataset\go_emotions_dataset.csv')
	last_count = 0

	@staticmethod
	def make_examples(num_of_batches, examples_per_batch):  # [[[][]][][]]
		model = Dataset.get_model()

		vector = []
		for i in range(num_of_batches):
			vector.append([])
			inner_vector = vector[i]
			for j in range(examples_per_batch):
				inner_vector.append([])
				inner_vector2 = inner_vector[j]

				while Dataset.df.example_very_unclear[Dataset.last_count] == 'TRUE':
					Dataset.last_count += 1

				# sentence
				text = Dataset.df.text[Dataset.last_count]

				regex = re.compile(r'[^a-zA-Z\s]')
				text = regex.sub('', text)
				text = text.lower()

				# sentence => array of words
				arr = text.split(" ")

				we_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				res = []

				# words => word embedding
				for word in arr:
					if(
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

				# 28 feelings vector res
				res.append(Dataset.df.admiration[Dataset.last_count])
				res.append(Dataset.df.amusement[Dataset.last_count])
				res.append(Dataset.df.anger[Dataset.last_count])
				res.append(Dataset.df.annoyance[Dataset.last_count])
				res.append(Dataset.df.approval[Dataset.last_count])
				res.append(Dataset.df.caring[Dataset.last_count])
				res.append(Dataset.df.confusion[Dataset.last_count])
				res.append(Dataset.df.curiosity[Dataset.last_count])
				res.append(Dataset.df.desire[Dataset.last_count])
				res.append(Dataset.df.disappointment[Dataset.last_count])
				res.append(Dataset.df.disapproval[Dataset.last_count])
				res.append(Dataset.df.disgust[Dataset.last_count])
				res.append(Dataset.df.embarrassment[Dataset.last_count])
				res.append(Dataset.df.excitement[Dataset.last_count])
				res.append(Dataset.df.fear[Dataset.last_count])
				res.append(Dataset.df.gratitude[Dataset.last_count])
				res.append(Dataset.df.grief[Dataset.last_count])
				res.append(Dataset.df.joy[Dataset.last_count])
				res.append(Dataset.df.love[Dataset.last_count])
				res.append(Dataset.df.nervousness[Dataset.last_count])
				res.append(Dataset.df.optimism[Dataset.last_count])
				res.append(Dataset.df.pride[Dataset.last_count])
				res.append(Dataset.df.realization[Dataset.last_count])
				res.append(Dataset.df.relief[Dataset.last_count])
				res.append(Dataset.df.remorse[Dataset.last_count])
				res.append(Dataset.df.sadness[Dataset.last_count])
				res.append(Dataset.df.surprise[Dataset.last_count])
				res.append(Dataset.df.neutral[Dataset.last_count])

				arr_final = []

				inner_vector2.append([we_arr])
				inner_vector2.append([res])

		return vector

	@staticmethod
	def get_model():

		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

		# use pre-trained model and use it
		model = downloader.load('glove-twitter-25')

		return model