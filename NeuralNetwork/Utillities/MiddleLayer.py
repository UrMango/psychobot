from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np


class WeightLayer(Layer):
	def __init__(self, output_size, std,  _id, _nudges_id ,_inputs_id, _outputs_id, num_of_inputs):
		super().__init__()
		mean = 0
		self.input = None
		self.nudges_id = _nudges_id
		self.inputs_id = _inputs_id
		self.outputs_id = _outputs_id
		self.id = _id
		self.weights = []
		self.num_of_inputs = []
		self.output_size = output_size
		for i in num_of_inputs:
			self.weights.append(np.random.normal(mean, std, (self.hidden_units, self.input_units)))
		self.bias = np.random.normal(mean, std, (1, output_size))
		self.type = LayerType.MIDDLE

	def forward_propagation(self, output_layers_dict, time):
		self.input = []

		for input_id in self.inputs_id:
			self.input.append(output_layers_dict[input_id] + time)
		self.output = np.zeros((1, self.output_size), dtype=np.float32)
		for i in self.num_of_inputs:
			self.output = np.add(np.dot(self.input[i], self.weights[i]), self.output) # sum up all the inputs
		self.output = np.add(self.output, self.bias)
		if time != -1:
			output_layers_dict[self.id+time] = self.output
		else:
			output_layers_dict[self.id] = self.output

		self.input = []
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict, time):
		self.input = []
		for input_id in self.inputs_id:
			self.input.append(output_layers_dict[input_id] + time)

		output_nudge = np.zeros((1, self.output_size), dtype=np.float32)
		if time != -1:
			for output_id in self.outputs_id:
				output_nudge = np.add(output_nudge, nudge_layers_dict[output_id+time]) #sum up all the nudges
		else:
			for output_id in self.outputs_id:
				output_nudge = np.add(output_nudge, nudge_layers_dict[output_id]) #sum up all the nudges but only for one time
		input_nudge = []
		weights_nudge = []
		for i in range(self.num_of_inputs):
			input_nudge.append(np.dot(output_nudge, self.weights[i].T))
			weights_nudge = np.append(np.dot(self.input[i].T, output_nudge))
		bias_nudge = output_nudge 												#define all the nudges

		for i in range(self.num_of_inputs):
			nudge_layers_dict[self.nudges_id[i] + time] += input_nudge[i]
		for i in range(self.num_of_inputs):
			nudge_layers_dict[self.nudges_id[i+len(self.num_of_inputs)]] += weights_nudge[i]
		nudge_layers_dict[self.nudges_id[i+2*len(self.num_of_inputs)]] += bias_nudge			#put all the nudges in the dict
		self.input = []
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate):
		for i in range(self.num_of_inputs):
			self.weights[i] -= nudge_layers_dict[self.nudges_id[i+len(self.num_of_inputs)]] * learning_rate
		self.bias -= nudge_layers_dict[self.nudges_id[i+2*len(self.num_of_inputs)]] * learning_rate
		return
