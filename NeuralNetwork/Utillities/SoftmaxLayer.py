from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np

class SoftmaxLayer(Layer):
	def __init__(self, _function, _derivative, _id, _inputs_id):
		super().__init__()
		self.input = None
		self.inputs_id = _inputs_id
		self.outputs_id = None
		self.function = _function
		self.derivative = _derivative
		self.id = _id
		self.type = LayerType.SOFTMAX

	def forward_propagation(self, output_layers_dict):
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id]
		self.output = self.function(self.input)  # making the activation
		output_layers_dict[self.id] = self.output
		self.input = None
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict, sentence_labels):
		loss = 0
		accuracy = 0
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id]
		output = output_layers_dict[self.id]
		output_error = np.zeros((1, len(output)), dtype=np.float32)

		for i in range(len(sentence_labels)):
			loss += -np.log(output[i])
			if sentence_labels[i] == 1:  # check how to add this value to the vector
				for k in range(len(output_error)):
					output_error[k] += self.derivative(output[i])  # output[i]-1

		for i, num in enumerate(output):
			if num > max_value:
				true_feeling_index = i
				max_value = num
		if sentence_labels[true_feeling_index] == 1:  # he predicted right
			accuracy = 1

		nudge_layers_dict["d" + self.input] = output_error
		self.input = None
		return nudge_layers_dict, loss, accuracy


	def nudge(self, nudge_layers_dict, learning_rate):
		pass
