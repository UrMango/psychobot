import numpy as np

def cost(current_output, right_output):
	return np.mean(np.power(current_output - right_output, 2))

def derivative_cost(current_output, right_output):
	return (2 / right_output.size) * (current_output-right_output)
