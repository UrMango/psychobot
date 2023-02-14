from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np


class MiddleLayer(Layer):
	def __init__(self, input_units, output_size, std,  _id, _inputs_id, set_parameters, parameters):
		super().__init__()
		mean = 0
		self.input = None
		self.inputs_id = _inputs_id
		self.id = _id

		self.m_weights = []
		self.v_weights = []
		self.m_bias = None
		self.v_bias = None

		self.weights = []
		self.num_of_inputs = len(input_units)
		self.output_size = output_size
		if set_parameters:
			for i in range(self.num_of_inputs):
				if self.inputs_id[i][-1] == "-":
					key = self.id + self.inputs_id[i][:-1] + "w"
				else:
					key = self.id + self.inputs_id[i] + "w"
				self.weights.append(parameters[key])
			self.bias = parameters[self.id + "b"]
		else:
			for i in range(self.num_of_inputs):
				self.weights.append(np.random.normal(mean, std, (input_units[i], output_size)))
				self.m_weights.append(np.zeros((input_units[i], output_size)))
				self.v_weights.append(np.zeros((input_units[i], output_size)))
			self.bias = np.random.normal(mean, std, (1, output_size))
			self.m_bias = np.zeros((1, output_size))
			self.v_bias = np.zeros((1, output_size))
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
			# if self.input[i].shape[1] != self.weights[i].shape[0]:
			# 	self.input[i] = self.input[i].T
			# print(self.id)
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
			if self.inputs_id[i][-1] == "-":
				key = "d" + self.inputs_id[i][:-1] + str(t-1)
			else:
				key = "d" + self.inputs_id[i] + time

			if key not in nudge_layers_dict.keys():
				nudge_layers_dict[key] = input_nudge[i]
			else:
				nudge_layers_dict[key] += input_nudge[i]
		for i in range(self.num_of_inputs):
			if self.inputs_id[i][-1] == "-":
				key = "d" + self.id + self.inputs_id[i][:-1] + "w"
			else:
				key = "d" + self.id + self.inputs_id[i] + "w"
			if key not in nudge_layers_dict.keys():
				nudge_layers_dict[key] = weights_nudge[i]
			else:
				nudge_layers_dict[key] += weights_nudge[i]
		key = "d" + self.id + "b"
		if key not in nudge_layers_dict.keys():
			nudge_layers_dict[key] = bias_nudge
		else:
			nudge_layers_dict[key] += bias_nudge		# put all the nudges in the dict
		self.input = []
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate, beta1, beta2, epsilon, batch_len):
		mhat_weights = []
		vhat_weights = []

		for i in range(self.num_of_inputs):
			if self.inputs_id[i][-1] == "-":
				key = "d" + self.id + self.inputs_id[i][:-1] + "w"
			else:
				key = "d" + self.id + self.inputs_id[i] + "w"
			self.m_weights[i] = beta1 * self.m_weights[i] + (1-beta1) * nudge_layers_dict[key]
			self.v_weights[i] = beta2 * self.v_weights[i] + (1-beta2) * np.power(nudge_layers_dict[key], 2) 

			mhat_weights.append(self.m_weights[i] * (1 / (1-beta1)))
			vhat_weights.append(self.v_weights[i] * (1 / (1-beta2)))

			self.weights[i] -= learning_rate * (1/batch_len) * np.divide(mhat_weights[i], (np.sqrt(vhat_weights[i])+epsilon))
		self.m_bias = beta1 * self.v_bias + (1 - beta1) * nudge_layers_dict["d" + self.id + "b"]
		self.v_bias = beta2 * self.v_bias + (1 - beta2) * np.power(nudge_layers_dict["d" + self.id + "b"], 2)

		mhat_bias = self.m_bias * (1 / (1 - beta1))
		vhat_bias = self.v_bias * (1 / (1 - beta2))

		self.bias -= learning_rate * (1 / batch_len) * np.divide(mhat_bias, (np.sqrt(vhat_bias) + epsilon))
		return

	def save_parameters(self, parameters_dict):
		for i in range(self.num_of_inputs):
			if self.inputs_id[i][-1] == "-":
				key = self.id + self.inputs_id[i][:-1] + "w"
			else:
				key = self.id + self.inputs_id[i] + "w"
			parameters_dict[key] = self.weights[i]
		parameters_dict[self.id + "b"] = self.bias
		return parameters_dict
