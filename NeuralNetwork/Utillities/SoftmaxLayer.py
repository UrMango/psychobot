from NeuralNetwork.Utillities.Layer import Layer, LayerType
from NeuralNetwork.Utillities.activation_functions import  Softmax

import numpy as np

class SoftmaxLayer(Layer):
	def __init__(self, _id, _inputs_id):
		super().__init__()
		self.input = None
		self.inputs_id = _inputs_id
		self.outputs_id = None
		self.function = Softmax.softmax
		self.id = _id
		self.type = LayerType.SOFTMAX

	def forward_propagation(self, output_layers_dict, t):
		time = str(t)
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id + time]
		self.output = self.function(self.input)  # making the activation
		output_layers_dict[self.id] = self.output
		self.input = None
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict, sentence_labels, t):
		time = str(t)
		accuracy = 0
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id+time]
		output = output_layers_dict[self.id]

		sentence_labels_vector = np.zeros((1, len(output[0])), dtype=np.float32)
		for i in range(len(sentence_labels)):
			sentence_labels_vector[0][i] = sentence_labels[i]

		output_nudge = np.subtract(output, sentence_labels_vector)

		log_soft_max = np.log(output)
		loss_vec = np.multiply(log_soft_max, sentence_labels_vector)
		loss = -np.sum(loss_vec)
		true_feeling_index = 0
		max_value = output[0][0]

		for i, num in enumerate(output[0]):
			if num > max_value:
				true_feeling_index = i
				max_value = num
		if sentence_labels[true_feeling_index] == 1:  # he predicted right
			accuracy = 1

		key = "d" + self.inputs_id[0] + time
		if key not in nudge_layers_dict.keys():
			nudge_layers_dict[key] = output_nudge
		else:
			nudge_layers_dict[key] += output_nudge

		self.input = None
		return nudge_layers_dict, loss, accuracy

	def nudge(self, nudge_layers_dict, learning_rate, batch_len):
		pass

	def save_parameters(self, parameters_dict):
		return parameters_dict
