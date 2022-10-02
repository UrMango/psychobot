import numpy as np

class Tanh:
	@staticmethod
	def tanh(x):
		return np.tanh(x)

	@staticmethod
	def tanh_derivative(x):
		return 1 - np.tanh(x) * np.tanh(x)

class Sig:
	@staticmethod
	def sigmoid(x):
		return 1 / (1+np.exp(-1*x))

	@staticmethod
	def inverse_sigmoid(x):
		return np.log(x/1-x)

	@staticmethod
	def derivative_sigmoid(x):
		return np.exp(-x) / pow((1+np.exp(-x)), 2)