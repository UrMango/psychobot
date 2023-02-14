from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np

class MultiplyLayer(Layer):
	def __init__(self, _id, _inputs_id):
		super().__init__()
		self.input = None
		self.inputs_id = _inputs_id
		self.id = _id
		self.type = LayerType.MULTIPLY

	def forward_propagation(self, output_layers_dict, t):

		time = str(t)
		self.input = []
		for input_id in self.inputs_id:
			if input_id[-1] == "-":
				self.input.append(output_layers_dict[input_id[:-1]+str(t-1)])
			else:
				self.input.append(output_layers_dict[input_id+time])
		self.output = np.multiply(self.input[0], self.input[1])  # assume there is only 2 vectors
		output_layers_dict[self.id+time] = self.output
		self.input = []
		return output_layers_dict


	def backward_propagation(self, nudge_layers_dict, output_layers_dict, t):
		self.input = []
		time = str(t)
		for input_id in self.inputs_id:
			if input_id[-1] == "-":
				self.input.append(output_layers_dict[input_id[:-1]+str(t-1)])
			else:
				self.input.append(output_layers_dict[input_id+time])
		output_nudge = nudge_layers_dict["d"+self.id+time]
		nudges = []
		nudges.append(np.multiply(output_nudge, self.input[1]))  # the nudge of the *first* input
		nudges.append(np.multiply(output_nudge, self.input[0]))  # the nudge of the *second* input
		i = 0
		for input_id in self.inputs_id:
			if input_id[-1] == "-":
				key = "d"+input_id[:-1]+str(t-1)
			else:
				key = "d"+input_id+time
			if key not in nudge_layers_dict.keys():
				nudge_layers_dict[key] = nudges[i]
			else:
				nudge_layers_dict[key] += nudges[i]
			i += 1

		self.input = []
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate, beta1, beta2, epsilon, batch_len):
		pass

	def save_parameters(self, parameters_dict):
		return parameters_dict
