import numpy as np
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh, Softmax

class OutputCell:
	@staticmethod
	def activate(activation_vector, parameters):
		# get hidden to output parameters
		how = parameters['how']

		# get outputs
		output_matrix = np.matmul(activation_vector, how)
		output_matrix = Softmax.softmax(output_matrix)

		return output_matrix

	@staticmethod
	def calculate_error(sentence_labels, output_cache, parameters):
		# to store the output errors for each time step
		output_error_cache = dict()
		activation_error_cache = dict()
		how = parameters['how']

		pred = output_cache['o']

		# calculate the output_error for time step 't'
		error_output = pred - sentence_labels

		# calculate the activation error for time step 't'
		error_activation = np.matmul(error_output, how.T)

		# store the output and activation error in dict
		output_error_cache['eo'] = error_output
		activation_error_cache['ea'] = error_activation

		return output_error_cache, activation_error_cache

	# calculate output cell derivatives
	@staticmethod
	def calculate_derivatives(output_error_cache, activation_cache, parameters):
		# to store the sum of derivatives from each time step
		dhow = np.zeros(parameters['how'].shape)

		batch_size = activation_cache['a1'].shape[0]

		# get output error
		output_error = output_error_cache['eo']

		# get input activation
		activation = activation_cache['a']

		# cal derivative and summing up!
		dhow += np.matmul(activation.T, output_error) / batch_size

		return dhow
