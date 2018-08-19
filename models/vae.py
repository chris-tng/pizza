from .nn import *
import torch as T
import numpy as np
from time import time

epsilon = 1e-10
gaussian_regularization = lambda mu, var: 0.5*(1. + T.log(var + epsilon) - mu.pow(2.) - var).sum()
bernoulli_likelihood = lambda x, x_hat: (x * T.log(x_hat + epsilon) + (1. - x) * T.log(1. - x_hat)).sum()
entropy = lambda px: -(px * T.log(px + epsilon)).sum()


def gumbel_softmax(pi, temp):
	"""
	Sample from Categorial distribution with parameter `pi` using 
	Gumbel softmax approximation [1]

	pi: torch.Tensor of dim (n_class, n_batch)

	[1] “Categorical Reparameterization by Gumbel-Softmax“. Eric Jang et al. 2017
	"""
	uniform_samples = T.rand_like(pi)
	s = (T.log(pi) - T.log(-T.log(uniform_samples))) / temp
	return softmax(s)



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
		t0 = time()
		for j in range(n_iterations+1):
			X = fetcher()
			loss = self.forward(X)
			# logistic
			hook(self, loss.item(), j, X.detach().numpy(), time()-t0)



class SSL_VAE(object):
	"""
	
	"""
	
	def __init__(self, params):
		self.__dict__.update(params)
		n_in, n_hidden, n_embedding, n_out = self.n_in, self.n_hidden, self.n_embedding, self.n_out 
		w_init, n_class = self.w_init, self.n_class
		
		# Encoder / Inference
		self.qy_x  = [DenseLayer(n_in, n_hidden, softplus, w_init), DenseLayer(n_hidden, n_class, softmax, w_init)]
		self.qz_yx = [DenseLayer(n_in+n_class, n_hidden, softplus, w_init), 
					  GaussianLayer(n_hidden, n_embedding, w_init)]
		
		# Decoder / Generative
		self.px_yz = [DenseLayer(n_class+n_embedding, n_hidden, softplus, w_init), 
					  DenseLayer(n_hidden, n_out, T.sigmoid, w_init)]
		
		self.optimizer(self.learning_rate, self.clamping, self.qy_x + self.qz_yx + self.px_yz)
		
	
	def forward(self, xu, xo, yo, train=True):
		qy_x, qz_yx, px_yz = self.qy_x, self.qz_yx, self.px_yz
		
		# Inference model
		n_unlabeled = xu.size()[1]
		n_labeled   = xo.size()[1]
		alpha = 0.1 *  n_unlabeled/n_labeled 
		# observed y - compute classification loss
		yo_pred = qy_x[1].forward(qy_x[0].forward(xo))
		classification_loss = alpha * bernoulli_likelihood(yo, yo_pred)
		
		# unobserved y - approximate samples
		yu_pred = qy_x[1].forward(qy_x[0].forward(xu))
		# temp = max(0.5, np.exp(-3e-5 * t)) updated every 2000 steps
		y_samples = gumbel_softmax(yu_pred, 0.5)
		
		y  = T.cat((yo, y_samples), dim=1); x  = T.cat((xo, xu), dim=1)  
		xy = T.cat((x, y), dim=0)
		mu_z, var_z = qz_yx[1].forward(qz_yx[0].forward(xy))
		z  = mu_z + T.sqrt(var_z) * T.rand_like(mu_z)
		
		# Generative model
		zy    = T.cat((y, z), dim=0)  
		x_hat = px_yz[1].forward(px_yz[0].forward(zy))
		loss  = (gaussian_regularization(mu_z, var_z) + bernoulli_likelihood(x, x_hat) \
				 + entropy(yu_pred)) / (n_unlabeled + n_labeled) + classification_loss / n_labeled
		if train:
			loss.backward()
			self.optimizer.optimize()
		return loss
	
	
	def train(self, n_iterations, fetcher, hook):
		t0 = time()
		for j in range(n_iterations+1):
			xu, xo, y = fetcher()
			loss = self.forward(xu, xo, y)
			# logistic
			hook(self, loss.item(), j, time()-t0)


