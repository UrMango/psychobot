from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np


class MiddleLayer(Layer):
	def __init__(self, input_units, output_size, std,  _id, _nudges_id, _inputs_id, num_of_inputs):
		super().__init__()
		mean = 0
		self.input = None
		self.nudges_id = _nudges_id
		self.inputs_id = _inputs_id
		self.id = _id
		self.weights = []
		self.num_of_inputs = num_of_inputs
		self.output_size = output_size
		for i in range(num_of_inputs):
			self.weights.append(np.random.normal(mean, std, (input_units[i], output_size)))
		self.bias = np.random.normal(mean, std, (1, output_size))
		self.type = LayerType.MIDDLE

	def forward_propagation(self, output_layers_dict, t):
		time = str(t)
		if t == -1:
			time = ""
		self.input = []

		for input_id in self.inputs_id:
			if input_id[-1] == "-":
				self.input.append(output_layers_dict[input_id[:-1] + str(t-1)])
			else:
				self.input.append(output_layers_dict[input_id + time])

		self.output = np.zeros((1, self.output_size), dtype=np.float32)
		for i in range(self.num_of_inputs):
			self.output += np.dot(self.input[i], self.weights[i])  # sum up all the inputs
		self.output = np.add(self.output, self.bias)
		output_layers_dict[self.id+time] = self.output

		self.input = []
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict, t):
		self.input = []
		time = str(t)
		for input_id in self.inputs_id:
			if input_id[-1] == "-":
				self.input.append(output_layers_dict[input_id[:-1] + str(t-1)])
			else:
				self.input.append(output_layers_dict[input_id + time])

		if t != -1:
			output_nudge = nudge_layers_dict["d"+self.id+time]
		else:
			output_nudge = nudge_layers_dict["d"+self.id]
		input_nudge = []
		weights_nudge = []
		for i in range(self.num_of_inputs):
			input_nudge.append(np.dot(output_nudge, self.weights[i].T))   # check why this is inverse
			weights_nudge.append(np.dot(self.input[i].T, output_nudge))
		bias_nudge = output_nudge 												# define all the nudges

		for i in range(self.num_of_inputs):
			key = None
			if self.nudges_id[i][-1] == "-":
				key = self.nudges_id[i][:-1] + str(t-1)
			else:
				key = self.nudges_id[i] + time
			if key not in nudge_layers_dict.keys():
				nudge_layers_dict[key] = input_nudge[i]
			else:
				nudge_layers_dict[key] += input_nudge[i]
		for i in range(self.num_of_inputs):
			key = self.nudges_id[i+self.num_of_inputs]
			if key not in nudge_layers_dict.keys():
				nudge_layers_dict[key] = weights_nudge[i]
			else:
				nudge_layers_dict[key] += weights_nudge[i]
		key = self.nudges_id[2*self.num_of_inputs]
		if key not in nudge_layers_dict.keys():
			nudge_layers_dict[key] = bias_nudge
		else:
			nudge_layers_dict[key] += bias_nudge		# put all the nudges in the dict
		self.input = []
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate, batch_len):
		for i in range(self.num_of_inputs):
			self.weights[i] -= nudge_layers_dict[self.nudges_id[i+self.num_of_inputs]] * learning_rate * (1/batch_len)
		self.bias -= nudge_layers_dict[self.nudges_id[2*self.num_of_inputs]] * learning_rate * (1/batch_len)
		return
