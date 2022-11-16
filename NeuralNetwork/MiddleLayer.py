from Layer import Layer, LayerType
import numpy as np
LEARNING_RATE = 0.1


class MiddleLayer(Layer):
	def __init__(self, input_size, output_size):
		self.weights = np.random.rand(input_size, output_size) - 0.5
		self.bias = np.random.rand(1, output_size) - 0.5
		self.type = LayerType.MIDDLE

	def forward_propagation(self, input_data):
		self.input = input_data
		self.output = np.dot(self.input, self.weights) + self.bias
		return self.output

	def backward_propagation(self, output_nudge):
		input_nudge = np.dot(output_nudge, self.weights.T)
		weights_nudge = np.dot(self.input.T, output_nudge)
		bias_nudge = output_nudge

		self.weights -= weights_nudge * LEARNING_RATE
		self.bias -= bias_nudge * LEARNING_RATE

		return input_nudge
