import numpy as np
import torch as T

def int_to_binary(d, length=8):
	"""
	Binarize an integer d to a list of 0 and 1. Length of list is fixed by `length`
	"""
	d_bin = '{0:b}'.format(d)
	d_bin = (length - len(d_bin)) * '0' + d_bin  # Fill in with 0
	return [int(i) for i in d_bin]


class Adagrad(object):
	
	def __call__(self, learning_rate=0.01, clamping=None, layers=None):
		self.learning_rate = learning_rate
		self.clamping      = clamping
		self.layers        = layers
		self.weights, self.cache = [], []
		# additional logic on top of `layers` attribute - code smell?
		for layer in layers:
			self.add(layer)
		
	
	def add(self, layer):
		"""
		Add a layer into Adagrad to keep track of and update its parameters
		"""
		self.weights.extend(layer.weights)
		self.cache.extend([T.zeros_like(w) for w in layer.weights])
		
				
	def optimize(self):
		for w, mw, i in zip(self.weights, self.cache, range(len(self.cache))):
			mw += w.grad.data.pow(2.)
			if self.clamping: w.grad.data.clamp_(-self.clamping, self.clamping)
			w.data += self.learning_rate * w.grad.data/ T.sqrt(mw + 1e-10)
			w.grad.zero_()
			
			