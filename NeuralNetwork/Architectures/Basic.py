import numpy as np
from NeuralNetwork.Architectures import Architecture
from NeuralNetwork.Utillities import Cost

ArchitectureType = Architecture.ArchitectureType
Architecture = Architecture.Architecture


class Basic(Architecture):
	# Constructor
	def __init__(self):
		super().__init__(ArchitectureType.BASIC)

	def run_model(self, input_data, layers):
		input_data_ = input_data.copy()
		for layer in layers:
			input_data_ = layer.forward_propagation(input_data_)
		return input_data_

	def train(self, examples, layers):  # [[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]],[[1,2,3,4],[1,2]]]
		# example_nudge = []
		# all_nudges = []
		for example in examples:
			current_output = self.run_model(np.array(example[0]), layers)
			nudge = Cost.derivative_cost(current_output, np.array(example[1]))
			for layer in reversed(layers):
				nudge = layer.backward_propagation(nudge)
		"""
			   if layer.type == LayerType.MIDDLE:
					example_nudge.append(nudge[1])
					example_nudge.append(nudge[2])
				nudge = nudge[0]
			example_nudge.reverse()
			all_nudges.append(example_nudge)

		average_nudge = np.dot(0, example_nudge.copy())
		for example in all_nudges:
			average_nudge = np.add(average_nudge, example)

		num_of_examples = 1 / len(all_nudges)
		average_nudge = np.dot(num_of_examples, average_nudge)
		i = 0
		for layer in self.layers:
			if layer.type == LayerType.MIDDLE:
				layer.bias -= average_nudge[i] * MiddleLayer.LEARNING_RATE
				i += 1
				layer.weights -= average_nudge[i] * MiddleLayer.LEARNING_RATE
				i += 1
		"""
