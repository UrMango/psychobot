import numpy as np
from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh, Softmax


class OutputCell:
	@staticmethod
	def activate(hidden, parameters):
		# get outputs
		output = np.add(np.matmul(parameters['ow'], hidden), parameters['ob'])
		softmax = Softmax.softmax(output)

		return output, softmax

	@staticmethod
	def calculate_error(sentence_labels , hidden_cache, output, softmax, parameters, loss, accuracy):
		# to store the output errors for each time step
		output_error = np.zeros([len(output)], dtype=np.float32)
		loss.append(0)
		for i in range(len(sentence_labels)):
			if sentence_labels[i] == 1: # check how to add this value to the vector
				loss[-1] += -np.log(softmax[i])  # add to loss the loss function
				for k in range(len(output_error)):
					output_error[k] += Softmax.derivative_softmax_and_log_by_func(softmax[i]) #softmax[i]-1
		true_feeling_index = 0
		max_value = output[0]
		for i, num in enumerate(output):
			if num > max_value:
				true_feeling_index = i
				max_value = num
		if sentence_labels[true_feeling_index] == 1: #he predicted right
			accuracy.append(1)
		else:
			accuracy.append(0)
		output_weights = parameters['ow']
		output_weights_error = np.matmul(np.reshape(output_error, (len(output_error), 1)), np.reshape(hidden_cache[-1], (len(hidden_cache[-1]), 1)).T)
		output_biases_error = output_error
		hidden_error = np.matmul(output_weights.T, output_error)

		return output_weights_error, output_biases_error, hidden_error, loss, accuracy


