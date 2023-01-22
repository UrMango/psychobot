from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np

class ActivationLayer(Layer):
	def __init__(self, _function, _derivative, _id, _inputs_id):
		super().__init__()
		self.input = None
		self.inputs_id = _inputs_id
		self.function = _function
		self.derivative = _derivative
		self.id = _id
		self.type = LayerType.ACTIVATION

	def forward_propagation(self, output_layers_dict, t):
		time = str(t)
		for input_id in self.inputs_id:  # relate on the fact that activation don't use h-1
			self.input = output_layers_dict[input_id+time]
		self.output = self.function(self.input)  # making the activation
		output_layers_dict[self.id+time] = self.output
		self.input = None
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict, t):
		time = str(t)
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id+time]
		output_nudge = nudge_layers_dict["d"+self.id+time]
		nudge_layers_dict["d" + self.inputs_id+time] = self.derivative(output_layers_dict[self.id + time])*output_nudge
		self.input = None
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate):
		pass
