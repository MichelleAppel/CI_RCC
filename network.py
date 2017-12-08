import numpy as np
import random

class Network():

	def __init__(self, weights_matrix=None, input_dim=23, output_dim=3, hidden_layer_dim = (6,), fitness=-1.): 	# Initial values
		self.dimensions = (input_dim,) + hidden_layer_dim + (output_dim,) 					# The dimensions of the weight matrix
		if weights_matrix == None:
			self.weights_matrix=self.generate_random_weights()
		else:
			self.weights_matrix = weights_matrix 								# Generate a random weight matrix
		self.fitness = fitness

	# Returns the sigmoid of a value
	def sigmoid(self, x):
		return 1/(1+np.e**(-x))

	def tanh(self, x):
		return np.tanh(x)

	# Returns the activation of a neuron
	def activation(self, W, A): 
		sum = 0 
		for w, a in zip(W, A):
			sum += w*a
		return self.tanh(sum)

	def layer_activations(self, weights_matrix, input_vec):
		activations = []
		for weights_vec in weights_matrix:
			activations.append(self.activation(weights_vec, input_vec))
		activations.append(1) # Add bias node
		return activations

	def total_activations(self, input_vec):
		input_vec = np.append(input_vec, [1]) # Add bias node
		activations = [input_vec]
		for weights_vec in self.weights_matrix:
			activations.append(self.layer_activations(weights_vec, activations[-1]))
		return activations[-1][:-1]

	def generate_random_weights(self):
		weights_matrix = []
		for i in range(len(self.dimensions)-1): # Layers
			weights_vec = []
			for n_nodes_1 in range(self.dimensions[i+1]): # Nodes of next layer
				weights = []
				for n_nodes in range(self.dimensions[i]): # Nodes of this layer
					weights.append(random.uniform(-1,1)) # Random weight between -1 and 1
				weights_vec.append(weights)
			weights_matrix.append(weights_vec)
		return weights_matrix

	def NN(self, input_vector):
		if len(input_vector) != self.dimensions[0]:
			raise ValueError('Input vector dimension is incorrect.')
		return self.total_activations(input_vector)

