import numpy as np

from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh, Softmax


class Cell:
	@staticmethod
	def activate(word, previous_hidden, parameters):
		hidden = np.add(np.add(np.matmul(parameters['iw'], word), np.matmul(parameters['hw'], previous_hidden)), parameters['hb'])

		# we're doing hidden equals to tanh of the hidden
		hidden = Tanh.tanh(hidden)
		return hidden

	@staticmethod
	def calculate_error(input, hidden_error, curr_hidden, before_hidden, parameters):
		hidden_raw_error = np.multiply(Tanh.tanh_derivative_by_func(curr_hidden), hidden_error)

		hidden_weights = parameters["hw"]

		input_weights_error = np.matmul(np.reshape(hidden_raw_error, (len(hidden_raw_error), 1)), np.reshape(input, (len(input), 1)).T)
		hidden_weights_error = np.matmul(np.reshape(hidden_raw_error, (len(hidden_raw_error), 1)), np.reshape(before_hidden, (len(before_hidden), 1)).T)
		hidden_biases_error = hidden_raw_error
		hidden_error = np.matmul(hidden_weights.T, hidden_raw_error)

		return hidden_error, input_weights_error, hidden_weights_error, hidden_biases_error

