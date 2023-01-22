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
		self.output = self.function(self.input) #making the activation
		output_layers_dict[self.id] = self.output
		self.input = None
		return output_layers_dict

	def backward_propagation(self, nudge_layers_dict, output_layers_dict):
		for input_id in self.inputs_id:
			self.input = output_layers_dict[input_id]
		nudge_layers_dict["d" + self.id] = self.derivative(self.output) #defind the gradiant of the softmax and the loss
		self.input = None
		return nudge_layers_dict

	def nudge(self, nudge_layers_dict, learning_rate):
		pass
