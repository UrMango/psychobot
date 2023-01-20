import numpy as np
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh, Softmax

class OutputCell:
	@staticmethod
	def activate(hidden, parameters):
		# get outputs
		output = np.add(np.matmul(parameters['wo'], hidden), parameters['bo'])
		softmax = Softmax.softmax(output)

		return output, softmax

	@staticmethod
	def calculate_error(sentence_labels,hidden_cache, output , softmax, parameters, loss):
		# to store the output errors for each time step
		output_error = np.zeros([len(output)], dtype=np.float32)
		for i in range(len(sentence_labels)):
			if sentence_labels[i] == 1: 		#check how to add this value to the vector
				for item in output_error:
					loss[-1] += -np.log(softmax[i])  #add to loss the loss function
					item += Softmax.derivative_softmax_and_log_by_func(softmax([i]))


		output_weights = parameters['wo']

		output_weights_error = np.matmul(output_error, hidden_cache[-1].T)
		output_biases_error = output_error
		hidden_error = np.matmul(output_weights.T, output_error)

		return output_weights_error, output_biases_error, hidden_error, loss

	# calculate output cell derivatives
	@staticmethod
	def calculate_derivatives(output_error_cache, activation_cache, parameters):
		# to store the sum of derivatives from each time step
		dhow = np.zeros(parameters['how'].shape)

		batch_size = activation_cache['a1'].shape[0]

		# get output error
		output_error = np.matrix(output_error_cache['eo'])

		# get input activation
		activation = np.matrix(activation_cache['a1'])
		# print(activation.shape)
		# print(output_error.shape)
		# cal derivative and summing up!
		ad = np.matmul(activation.T, output_error)
		dhow += np.matmul(activation.T, output_error) / batch_size

		return dhow
