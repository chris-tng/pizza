from .nn import *
import torch as T
import numpy as np

gaussian_regularization = lambda mu, var: 0.5*(1. + T.log(var + 1e-10) - mu.pow(2.) - var).sum()
bernoulli_likelihood = lambda x, x_hat: (x * T.log(x_hat + 1e-10) + (1. - x) * T.log(1. - x_hat)).sum()

class VAE(object):
	"""
	Vanilla VAE with one hidden layer and one embedding layer
	"""

	def __init__(self, params):
		self.__dict__.update(params)
		n_in, n_hidden, n_embedding, n_out = self.n_in, self.n_hidden, self.n_embedding, self.n_out
		# Encoder
		self.encoder_x = DenseLayer(n_in, n_hidden, T.tanh)
		self.encoder_z = GaussianLayer(n_hidden, n_embedding)
		
		# Decoder
		self.decoder_z = DenseLayer(n_embedding, n_hidden, T.tanh)
		self.decoder_x = DenseLayer(n_hidden, n_out, T.sigmoid)
		
		self.optimizer(self.learning_rate, self.clamping, [self.encoder_x, self.encoder_x, 
														   self.decoder_z, self.decoder_x])
		
		
	def forward(self, X, train=True):
		h_en = self.encoder_x.forward(X)
		mu_z, var_z = self.encoder_z.forward(h_en)
		epsilon = 1e-10
		n_batch = X.size()[1]
		# sample from z
		e = T.randn_like(mu_z)
		z = mu_z + T.sqrt(var_z) * e
		
		h_de  = self.decoder_z.forward(z)
		x_hat = self.decoder_x.forward(h_de)
		loss  = (gaussian_regularization(mu_z, var_z) + bernoulli_likelihood(X, x_hat)) / n_batch

		if train:
			loss.backward()
			self.optimizer.optimize()
		return loss
	
	
	def train(self, n_iterations, fetcher, hook):
		for j in range(n_iterations+1):
			X = fetcher()
			loss = self.forward(X)
			# logistic
			hook(self, loss.item(), j, X.detach().numpy())

