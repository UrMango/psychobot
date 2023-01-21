from NeuralNetwork.Utillities.Layer import Layer, LayerType
import numpy as np

class ActivationLayer(Layer):
	def __init__(self, function, derivative, id):
		super().__init__()
		self.function = function
		self.derivative = derivative
		self.id = id
		self.type = LayerType.ACTIVATION

	def forward_propagation(self, list_of_inputs,dict):
		self.input = list_of_inputs[0]
		self.output = self.function( list_of_inputs[0])
		return self.output

	def backward_propagation(self, output_nudge,dict):
		return self.derivative(self.input)*output_nudge

	def nudge(self, dict):
		pass
