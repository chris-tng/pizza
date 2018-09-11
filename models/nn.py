import torch as T
import numpy as np

# Common activations
softplus = lambda x: T.log(1. + T.exp(x))
softmax  = lambda x: T.exp(x) / T.exp(x).sum(dim=0)  # x of dim (n_x, n_batch)


class NNLayer(object):
	"""
	Base class for Neural network layer
	"""

	def __init__(self, n_out, n_in, w_init=(None, None), device=T.device('cpu')):
		self.n_in, self.n_out, self.w_init, self.dtype, self.device = n_in, n_out, w_init, T.float64, device
		# subclass needs to implement weight init
		
	
	def _w_init(self, n_out, n_in):
		dtype, device, init_type, scale = self.dtype, self.device, self.w_init[0], self.w_init[1]
		if init_type == 'gaussian':  
			# N(0, scale)
			return (T.randn(n_out, n_in, dtype=dtype, device=device) * scale).requires_grad_(), \
					T.zeros(n_out, 1, requires_grad=True, dtype=dtype, device=device)
		elif init_type == 'uniform': 
			# Unif(-scale, scale)
			return ((2*T.rand(n_out, n_in, dtype=dtype, device=device) - 1.) * scale).requires_grad_(), \
					T.zeros(n_out, 1, requires_grad=True, dtype=dtype, device=device)	
		else:
			return (T.randn(n_out, n_in, dtype=dtype, device=device) * 1./np.sqrt(n_in)).requires_grad_(), \
					T.zeros(n_out, 1, requires_grad=True, dtype=dtype, device=device)
				
		
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
		return "<{0} shape={1} activation='{2}'>".format(self.__class__.__name__, self.shape, self.activation.__name__)

	
	def __str__(self):
		return self.__repr__()

		


class DropoutLayer(NNLayer):

	def __init__(self, n_out, n_in, dropout, device):
		super().__init__(n_out, n_in, device=device)
		self.dropout = dropout


	def forward(self, x):
		dropout, dtype, device = self.dropout, self.dtype, self.device
		mask = T.tensor(np.random.binomial(1, 1.-dropout, tuple(x.size()))/(1.-dropout), dtype=dtype, device=device)
		return mask * x			




class LSTMLayer(NNLayer):
	"""
	LSTM has the following structure

	(n_out) [  ]
	h ----> [  ]   
		    [  ]
		     ^
		     | x  (n_in)
	"""
	
	def __init__(self, n_out, n_in, w_init=(None, None), device=T.device('cpu'), trainable_init=True):
		super().__init__(n_out, n_in, w_init, device)
		# initialize the forget bias to large value to remember more
		self.bf = T.ones(n_out, 1, requires_grad=True, dtype=self.dtype, device=device)
		self.Wf, _       = self._w_init(n_out, n_in + n_out) 
		self.Wi, self.bi = self._w_init(n_out, n_in + n_out)
		self.Wg, self.bg = self._w_init(n_out, n_in + n_out)
		self.Wo, self.bo = self._w_init(n_out, n_in + n_out)
		self.h0, self.c0 = self._w_init(n_out, 1)[1], self._w_init(n_out, 1)[1] \
		if trainable_init else T.zeros(n_out, 1, dtype=dtype, device=device), T.zeros(n_out, 1, dtype=dtype, device=device)

	
	def forward(self, x, hprev=None, cprev=None):
		"""
		Inputs:
		- x: torch.Tensor of dim (n_x, n_batch)
		- hprev: torch.Tensor of dim (n_out, 1)
		- cprev: torch.Tensor of dim (n_out, 1)
		
		hprev, cprev are hidden states from previous time step. It could happen in the case of character-level model where 
		the input is a stream of characters. We use a sliding window of size `n_batch` to feed that window of input into LSTM,
		then hidden states are preserved and forwarded to the next window.
		
		Empty memory occurs in case of `learning to sum bits` where each sliding window is a separate training case.
		
		Returns:
		- hs: list of torch.Tensor as outputs from the upper layer
		"""
		n_in, n_out    = self.n_in, self.n_out
		Wi, Wf, Wo, Wg = self.Wi, self.Wf, self.Wo, self.Wg   
		bi, bf, bo, bg = self.bi, self.bf, self.bo, self.bg  
		
		hprev = self.h0 if hprev is None else hprev 
		cprev = self.c0 if cprev is None else cprev 
		
		_hs, self.cs, self.gate_is, self.gate_fs, self.gate_gs, self.gate_os = [], [], [], [], [], []
		n_batch = x.size()[1]
		for j in range(n_batch):
			_x = x[:, j].view(-1, 1)
			xh = T.cat((_x, hprev), dim=0)
			gate_i = T.sigmoid(Wi.mm(xh) + bi)
			gate_f = T.sigmoid(Wf.mm(xh) + bf)
			gate_g = T.sigmoid(Wg.mm(xh) + bg)
			gate_o = T.tanh(Wo.mm(xh) + bo)
			c = gate_f * cprev + gate_i * gate_g
			h = gate_o * T.tanh(c)
			hprev  = h
			cprev  = c
			_hs.append(h); self.cs.append(c); self.gate_is.append(i)
			self.gate_fs.append(f); self.gate_gs.append(g); self.gate_os.append(o)
		hs = T.cat(_hs, dim=1)
		return hs




class RNNLayer(NNLayer):
    
    def __init__(self, n_out, n_in, w_init=(None, None), device=T.device('cpu'), trainable_init=True):
        super().__init__(n_out, n_in, w_init, device)
        
        self.Wx, _       = self._w_init(n_out, n_in)
        self.Wh, self.bh = self._w_init(n_out, n_out)
        self.h0 = self._w_init(n_out, 1)[1] if trainable_init else T.tensor(n_out, 1, dtype=self.dtype, device=device)
    
    
    def forward(self, x, hprev=None):
        Wx, Wh, bh = self.Wx, self.Wh, self.bh
        n_batch = x.size()[1]
        
        _hs = []
        hprev = self.h0 if hprev is None else hprev
        for j in range(n_batch):
            h = T.tanh(Wx.mm(x[:, j].view(-1, 1)) + Wh.mm(hprev) + bh)
            hprev = h
            _hs.append(hprev)
        hs = T.cat(_hs, dim=1)
        return hs




class DenseLayer(NNLayer):
	
	def __init__(self, n_out, n_in, activation, w_init=(None, None), device=T.device('cpu')):
		super().__init__(n_out, n_in, w_init, device)
		self.activation = activation
		self.W, self.b  = self._w_init(n_out, n_in)
		
		
	def forward(self, x):
		"""
		X: torch.Tensor of dim (n_x, n_batch)
		"""		
		self.h = self.W.mm(x) + self.b
		return self.activation(h)
		



class EmbeddingLayer(NNLayer):
    """
    Layer acts as a trainable distributed representation or an embedding of dim (n_in, n_out)
    """
    
    def __init__(self, n_out, n_in, w_init=(None, None), device=T.device('cpu')):
        super().__init__(n_out, n_in, w_init, device)
        self.embedding, _ = self._w_init(n_out, n_in)
        
        
    def forward(self, x):
        """
        x: list of indexes
        """
        return self.embedding[x]




class GaussianLayer(NNLayer):
	"""
	Layer to model mean and var of Gaussian distribution, usually used in VAE
	"""
	def __init__(self, n_out, n_in, w_init=(None, None), device=T.device('cpu')):
		super().__init__(n_out, n_in, w_init, device)
		self.Wm, self.bm = self._w_init(n_out, n_in)
		self.Ws, self.bs = self._w_init(n_out, n_in)
		
	
	def forward(self, x):
		"""
		x: torch.Tensor of dim (n_x, n_batch)
		"""
		mu   = self.Wm.mm(x) + self.bm
		var  = T.exp(self.Ws.mm(X) + self.bs)
		return mu, var
		