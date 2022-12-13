import numpy as np

from NeuralNetwork.Utillities.activation_functions import Sigmoid, Tanh


class Cell():
	@staticmethod
	def activate(batch, prev_activation_mat, prev_cell_mat, params):
		# get parameters
		fgw = params['fgw']
		igw = params['igw']
		ogw = params['ogw']
		ggw = params['ggw']

		# concat batch data and prev_activation matrix
		concat_dataset = np.concatenate((batch, prev_activation_mat), axis=0)

		# forget gate activations
		fa = np.matmul(concat_dataset, fgw)
		fa = Sigmoid.sigmoid(fa)

		# input gate activations
		ia = np.matmul(concat_dataset, igw)
		ia = Sigmoid.sigmoid(ia)

		# output gate activations
		oa = np.matmul(concat_dataset, ogw)
		oa = Sigmoid.sigmoid(oa)

		# gate gate activations
		ga = np.matmul(concat_dataset, ggw)
		ga = Tanh.tanh(ga)

		# new cell memory matrix
		cell_memory_matrix = np.multiply(fa, prev_cell_mat) + np.multiply(ia, ga)

		# current activation matrix
		activation_vector = np.multiply(oa, Tanh.tanh(cell_memory_matrix))

		# lets store the activations to be used in back prop
		lstm_activations = dict()
		lstm_activations['fa'] = fa
		lstm_activations['ia'] = ia
		lstm_activations['oa'] = oa
		lstm_activations['ga'] = ga

		return lstm_activations, cell_memory_matrix, activation_vector

	@staticmethod
	def calculate_error(activation_output_error, next_activation_error, next_cell_error, parameters,
										 lstm_activation, cell_activation, prev_cell_activation):
		# activation error =  error coming from output cell and error coming from the next lstm cell
		activation_error = activation_output_error + next_activation_error

		# output gate error
		oa = lstm_activation['oa']
		eo = np.multiply(activation_error, Tanh.tanh(cell_activation))
		eo = np.multiply(np.multiply(eo, oa), 1 - oa)

		# cell activation error
		cell_error = np.multiply(activation_error, oa)
		cell_error = np.multiply(cell_error, Tanh.tanh_derivative(Tanh.tanh(cell_activation)))
		# error also coming from next lstm cell
		cell_error += next_cell_error

		# input gate error
		ia = lstm_activation['ia']
		ga = lstm_activation['ga']
		ei = np.multiply(cell_error, ga)
		ei = np.multiply(np.multiply(ei, ia), 1 - ia)

		# gate gate error
		eg = np.multiply(cell_error, ia)
		eg = np.multiply(eg, Tanh.tanh_derivative(ga))

		# forget gate error
		fa = lstm_activation['fa']
		ef = np.multiply(cell_error, prev_cell_activation)
		ef = np.multiply(np.multiply(ef, fa), 1 - fa)

		# prev cell error
		prev_cell_error = np.multiply(cell_error, fa)

		# get parameters
		fgw = parameters['fgw']
		igw = parameters['igw']
		ggw = parameters['ggw']
		ogw = parameters['ogw']

		# embedding + hidden activation error
		embed_activation_error = np.matmul(ef, fgw.T)
		embed_activation_error += np.matmul(ei, igw.T)
		embed_activation_error += np.matmul(eo, ogw.T)
		embed_activation_error += np.matmul(eg, ggw.T)

		input_hidden_units = fgw.shape[0]
		hidden_units = fgw.shape[1]
		input_units = input_hidden_units - hidden_units

		# prev activation error
		prev_activation_error = embed_activation_error[input_units:]

		# store lstm error
		lstm_error = dict()
		lstm_error['ef'] = ef
		lstm_error['ei'] = ei
		lstm_error['eo'] = eo
		lstm_error['eg'] = eg

		return prev_activation_error, prev_cell_error, lstm_error

	# calculate derivatives for single lstm cell
	@staticmethod
	def calculate_derivatives(lstm_error, activation_matrix, sentence_size):
		# get error for single time step
		ef = lstm_error['ef']
		ei = lstm_error['ei']
		eo = lstm_error['eo']
		eg = lstm_error['eg']

		# get input activations for this time step
		concat_matrix = activation_matrix

		batch_size = sentence_size

		# cal derivatives for this time step
		dfgw = np.matmul(concat_matrix.T, ef) / batch_size
		digw = np.matmul(concat_matrix.T, ei) / batch_size
		dogw = np.matmul(concat_matrix.T, eo) / batch_size
		dggw = np.matmul(concat_matrix.T, eg) / batch_size

		# store the derivatives for this time step in dict
		derivatives = dict()
		derivatives['dfgw'] = dfgw
		derivatives['digw'] = digw
		derivatives['dogw'] = dogw
		derivatives['dggw'] = dggw

		return derivatives
