import torch as T
import numpy as np


class NNLayer(object):

	def __init__(self, n_in, n_out, dropout=None):
		self.n_in, self.n_out, self.dropout, self.dtype = n_in, n_out, dropout, T.float64
		# subclass needs to implement weight init
		
	
	def _w_init(self, n_out, n_in):
		return (T.randn(n_out, n_in, dtype=self.dtype) * 1./np.sqrt(n_in)).requires_grad_(), \
				T.zeros(n_out, 1, requires_grad=True, dtype=self.dtype)

		
	def forward(self):
		pass
	
	
	def _w_names(self):
		"""
		Return a list of weight names
		"""
		return [k for k, v in self.__dict__.items() if hasattr(v, 'requires_grad') and v.requires_grad]
	
	
	@property
	def weights(self):
		return [self[w] for w in self._w_names()]
	
	
	@property
	def shape(self):
		return (self.n_in, self.n_out)
	
	
	def _n_params(self):
		"""
		Return #parameters of the layers
		"""
		prod = lambda a, b: a*b
		return sum(prod(*tuple(w.size())) for w in self.weights)
	
	
	def __getitem__(self, key):
		return getattr(self, key)
	
	
	def __setitem__(self, key, value):
		setattr(self, key, value)


	def __repr__(self):
		return "<{0} shape={1} non_linear='{2}'>".format(self.__class__.__name__, self.shape, self.non_linear.__name__)

	
	def __str__(self):
		return self.__repr__()
	
		


class LSTMLayer(NNLayer):
	
	def __init__(self, n_in, n_out, dropout=None):
		super().__init__(n_in, n_out, dropout)
		n_in, n_out, dropout = self.n_in, self.n_out, self.dropout
		self.non_linear = None
		# initialize the forget bias to large value to remember more
		self.bf = T.ones(n_out, 1, requires_grad=True, dtype=self.dtype)  
		self.Wf, _       = self._w_init(n_out, n_in + n_out) 
		self.Wi, self.bi = self._w_init(n_out, n_in + n_out)
		self.Wg, self.bg = self._w_init(n_out, n_in + n_out)
		self.Wo, self.bo = self._w_init(n_out, n_in + n_out)

	
	def forward(self, x, hprev=None, cprev=None):
		"""
		Inputs:
		- x: torch.Tensor of dim (n_x, n_batch)
		- hprev: torch.Tensor of dim (n_out, 1)
		- cprev: torch.Tensor of dim (n_out, 1)
		
		hprev, cprev are hidden states from previous time step. It could happen in the case of character-level model where 
		the input is a stream of characters. We use a sliding window of size `n_batch` to feed that window of input into LSTM,
		then hidden states are preserved and forwarded to the next window
		
		Empty memory occurs in case of `learning to sum bits` where each sliding window is a separate training case.
		
		Returns:
		- hs: list of torch.Tensor as outputs from the layer to be used in subsequent layer
		"""
		n_in, n_out, dtype, dropout = self.n_in, self.n_out, self.dtype, self.dropout
		Wi, Wf, Wo, Wg = self.Wi, self.Wf, self.Wo, self.Wg   
		bi, bf, bo, bg = self.bi, self.bf, self.bo, self.bg   
		
		hprev   = T.zeros(n_out, 1, dtype=dtype) if hprev == None else hprev
		cprev   = T.zeros(n_out, 1, dtype=dtype) if cprev == None else cprev
		_hs     = []
		n_batch = x.size()[1]
		for j in range(n_batch):
			_x = x[:, j].view(-1, 1)
			xh = T.cat((_x, hprev), dim=0)
			i  = T.sigmoid(Wi.mm(xh) + bi)
			f  = T.sigmoid(Wf.mm(xh) + bf)
			g  = T.sigmoid(Wg.mm(xh) + bg)
			o  = T.tanh(Wo.mm(xh) + bo)
			c  = f * cprev + i * g
			_h = o * T.tanh(c)
			# batch norm - maybe?
			h = _h * T.tensor(np.random.binomial(1, 1.-dropout, 
												 tuple(_h.size()))/(1.-dropout), dtype=dtype) if dropout else _h
			hprev  = h
			cprev  = c
			_hs.append(h)
		hs = T.cat(_hs, dim=1)
		return hs
	



class DenseLayer(NNLayer):
	
	def __init__(self, n_in, n_out, non_linear, dropout=None):
		super().__init__(n_in, n_out, dropout)
		n_in, n_out, dropout = self.n_in, self.n_out, self.dropout
		self.non_linear = non_linear
		self.W, self.b = self._w_init(n_out, n_in)
		
		
	def forward(self, X):
		"""
		X: torch.Tensor of dim (n_x, n_batch)
		"""
		dtype, dropout = self.dtype, self.dropout
		
		h = self.W.mm(X) + self.b
		_out = self.non_linear(h)
		# perform dropout - may switch this before non-linear for ReLU unit
		out = _out * T.tensor(np.random.binomial(1, 1.-dropout, 
												 tuple(_out.size()))/(1.-dropout), dtype=dtype) if dropout else _out
		return out
	   



class GaussianLayer(NNLayer):
	"""
	Layer to model mean and logvar of Gaussian 
	"""
	def __init__(self, n_in, n_out, dropout=None):
		super().__init__(n_in, n_out, dropout)
		n_in, n_out, dropout = self.n_in, self.n_out, self.dropout
		self.Wm, self.bm = self._w_init(n_out, n_in)
		self.Ws, self.bs = self._w_init(n_out, n_in)
		
	
	def forward(self, X):
		"""
		X: torch.Tensor of dim (n_x, n_batch)
		"""
		mu   = self.Wm.mm(X) + self.bm
		var  = T.exp(self.Ws.mm(X) + self.bs)
		mask = T.tensor(np.random.binomial(1, 1.-dropout, tuple(mu.size()))/(1.-dropout), dtype=dtype) if self.dropout else 1.
		# use the same dropout mask for both
		_mu  = mu * mask 
		_var = var * mask
		return _mu, _var
		