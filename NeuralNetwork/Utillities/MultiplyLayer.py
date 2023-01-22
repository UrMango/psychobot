from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np

class MultiplyLayer(Layer):
	def __init__(self, _nudges_id, _id, _inputs_id, _outputs_id):
		super().__init__()
		self.input = None
		self.inputs_id = _inputs_id
		self.outputs_id = _outputs_id
		self.nudges_id = _nudges_id
		self.id = _id
		self.type = LayerType.MULTIPLY

	def forward_propagation(self, output_layers_dict, time):
		self.input = []
		for input_id in self.inputs_id:
			self.input.append(output_layers_dict[input_id+time])
		self.output = np.multiply(self.input[0], self.input[1])
		output_layers_dict[self.id+time] = self.output
		self.input = []
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict, time):
		self.input = []
		for input_id in self.inputs_id:
			self.input.append(output_layers_dict[input_id+time])
		output_nudge = np.zeros((1, self.output_size), dtype=np.float32)
		for output_id in self.outputs_id:
			output_nudge = np.add(nudge_layers_dict[output_id+time], output_nudge) #sum up all the nudges
		nudges = []
		nudges.append(np.multiply(output_nudge, self.input[1])) # the nudge of the *first* input
		nudges.append(np.multiply(output_nudge, self.input[0])) # the nudge of the *second* input
		i = 0
		for id in self.nudges_id:
			nudge_layers_dict[id+time] = nudges[i]
			i += 1

		self.input = []
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate):
		pass
