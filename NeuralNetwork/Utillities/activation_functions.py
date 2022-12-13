import numpy as np

class Tanh:
	@staticmethod
	def tanh(x):
		return np.tanh(x)

	@staticmethod
	def tanh_derivative(x):
		return 1 - x * x

class Sigmoid:
	@staticmethod
	def sigmoid(x):
		return 1 / (1+np.exp(-1*x))

	@staticmethod
	def inverse_sigmoid(x):
		return np.log(x/1-x)

	@staticmethod
	def derivative_sigmoid(x):
		return np.exp(-x) / pow((1+np.exp(-x)), 2)

class Softmax:
	@staticmethod
	def softmax(x):
		return np.exp(x)/sum(np.exp(x))

	@staticmethod
	def inverse_softmax(x):
		return np.log(x / 1 - x)

	@staticmethod
	def derivative_softmax(x):
		return np.exp(-x) / pow((1 + np.exp(-x)), 2)
