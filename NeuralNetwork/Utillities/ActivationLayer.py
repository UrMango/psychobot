from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np

class ActivationLayer(Layer):
	def __init__(self, _function, _derivative, _id, _inputs_id, _outputs_id):
		super().__init__()
		self.input = None
		self.inputs_id = _inputs_id
		self.outputs_id = _outputs_id
		self.function = _function
		self.derivative = _derivative
		self.id = _id
		self.type = LayerType.ACTIVATION

	def forward_propagation(self, output_layers_dict, time):
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id+time]
		self.output = self.function(self.input) #making the activation
		output_layers_dict[self.id+time] = self.output
		self.input = None
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict, time):
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id+time]
		output_nudge = np.zeros((1, self.output_size), dtype=np.float32)
		for output_id in self.outputs_id:
			output_nudge = np.add(nudge_layers_dict[output_id+time], output_nudge) #sum up all the nudges
		nudge_layers_dict["d" + self.id+time] = self.derivative(self.output)*output_nudge #defind the gradiant of the layer and put it in the
		self.input = None
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate):
		pass
