import numpy as np


class Tanh:
	@staticmethod
	def tanh(x):
		return np.tanh(x)

	@staticmethod
	def tanh_derivative_by_func(tanh):
		return 1 - tanh * tanh


class Sigmoid:
	@staticmethod
	def sigmoid(x):
		return 1 / (1+np.exp(-1*x))

	@staticmethod
	def inverse_sigmoid(x):
		return np.log(x/1-x)

	@staticmethod
	def derivative_sigmoid_by_func(sig):
		return sig * (1-sig)


class Softmax:
	@staticmethod
	def softmax(x):
		x -= np.max(x)
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	@staticmethod
	def inverse_softmax(x):
		return np.log(x / 1 - x)

	@staticmethod
	def derivative_softmax_and_log_by_func(s):
		return s-1
