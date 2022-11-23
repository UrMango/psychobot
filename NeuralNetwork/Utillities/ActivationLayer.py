from Layer import Layer, LayerType
import numpy as np

class ActivationLayer(Layer):
	def __init__(self, function, derivative):
		self.function = function
		self.derivative = derivative
		self.type = LayerType.ACTIVATION

	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = self.function(input_data)
		return self.output

	def backward_propagation(self, output_nudge):
		return self.derivative(self.input)*output_nudge
