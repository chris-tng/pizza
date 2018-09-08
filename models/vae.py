from .nn import *
import torch as T
import numpy as np
from time import time

epsilon = 1e-10
gaussian_regularization = lambda mu, var: 0.5*(1. + T.log(var + epsilon) - mu.pow(2.) - var).sum()
bernoulli_likelihood = lambda x, x_hat: (x * T.log(x_hat + epsilon) + (1. - x) * T.log(1. - x_hat)).sum()
entropy = lambda px: -(px * T.log(px + epsilon)).sum()
gaussian_likelihood = lambda x, mu, var: -0.5*( (x - mu).pow(2.)/(var + epsilon) + T.log(var + epsilon) ).sum()
gaussian_entropy = lambda var: T.log(var + epsilon).sum()
categorial_regularization = lambda pi, n_class: -(pi*T.log(pi * n_class + epsilon)).sum() 


def rbf_kernel(x, y):
    """
    x: dim, n_x
    y: dim, n_y
    
    Returns:
    - K: n_x, n_y where K[i,j] is the RBF of x[:,i] and y[:,j] 
    """
    dim, n_x, n_y = x.size()[0], x.size()[1], y.size()[1]
    x_tiled = x.transpose(0, 1).reshape(n_x, 1, dim).expand(n_x, n_y, dim)
    y_tiled = y.transpose(0, 1).reshape(1, n_y, dim).expand(n_x, n_y, dim)
    return T.exp(-(x_tiled - y_tiled).pow(2.).mean(dim=2) / T.tensor(dim, dtype=T.float64))


def mmd(x, y):
    Kxx = rbf_kernel(x, x)
    Kyy = rbf_kernel(y, y)
    Kxy = rbf_kernel(x, y)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


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
		n_in, n_hidden, n_embedding, n_out, w_init, device = self.n_in, self.n_hidden, self.n_embedding, self.n_out, self.w_init, self.device
		hidden_nonlinear = self.hidden_nonlinear   # default T.tanh
		# Encoder
		self.encoder_x = DenseLayer(n_in, n_hidden, hidden_nonlinear, w_init, device)
		self.encoder_z = GaussianLayer(n_hidden, n_embedding, w_init, device)
		
		# Decoder
		self.decoder_z = DenseLayer(n_embedding, n_hidden, hidden_nonlinear, w_init, device)
		self.decoder_x = DenseLayer(n_hidden, n_out, T.sigmoid, w_init, device)
		
		self.optimizer(self.learning_rate, self.clamping, [self.encoder_x, self.encoder_x, 
														   self.decoder_z, self.decoder_x])
		
		
	def forward(self, X, train=True):
		h_en = self.encoder_x.forward(X)
		mu_z, var_z = self.encoder_z.forward(h_en)
		epsilon = 1e-10
		n_batch = X.size()[1]
		# sample from z
		z = mu_z + T.sqrt(var_z) * T.randn_like(mu_z)
		
		h_de  = self.decoder_z.forward(z)
		x_hat = self.decoder_x.forward(h_de)	
		# x_samples   = mu_x + T.sqrt(var_x) * T.rand_like(mu_x)  # for sampling purpose only
		loss  = (gaussian_regularization(mu_z, var_z) + bernoulli_likelihood(X, x_hat)) / n_batch
		#loss  = (gaussian_regularization(mu_z, var_z) + gaussian_likelihood(X, mu_x, var_x)) / n_batch

		if train:
			loss.backward()
			self.optimizer.optimize()
		return loss, {'z_samples': z.detach().numpy(), 'x_hat': x_hat.detach().numpy() }
					  #'mu_x': mu_x.detach().numpy(), 'var_x': var_x.detach().numpy()}
	
	
	def train(self, n_iterations, fetcher, hook):
		t0 = time()
		for j in range(n_iterations+1):
			X = fetcher()
			loss, results = self.forward(X)
			# logistic
			hook(self, loss.item(), j, time()-t0, results)



class SSL_VAE(object):
	"""
	Semi-supervised VAE as described in [2014 DP.Kingma et al]
	"""
	
	def __init__(self, params):
		self.__dict__.update(params)
		n_in, n_hidden, n_embedding, n_out = self.n_in, self.n_hidden, self.n_embedding, self.n_out 
		w_init, n_class, device = self.w_init, self.n_class, self.device
		
		# Encoder / Inference
		self.qy_x  = [DenseLayer(n_in, n_hidden, softplus, w_init, device), DenseLayer(n_hidden, n_class, softmax, w_init, device)]
		self.qz_yx = [DenseLayer(n_in+n_class, n_hidden, softplus, w_init, device), 
					  GaussianLayer(n_hidden, n_embedding, w_init, device)]
		
		# Decoder / Generative
		self.px_yz = [DenseLayer(n_class+n_embedding, n_hidden, softplus, w_init, device), 
					  DenseLayer(n_hidden, n_out, T.sigmoid, w_init, device)]
		
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



class AuxVAE(object):
    
    def __init__(self, params):
        self.__dict__.update(params)
        n_in, n_hidden, n_embedding, n_out = self.n_in, self.n_hidden, self.n_embedding, self.n_out
        n_aux, w_init, device = self.n_aux, self.w_init, self.device
        
        # Encoder / Inference
        self.qa_x  = [DenseLayer(n_in, n_hidden, T.relu, w_init, device), GaussianLayer(n_hidden, n_aux, w_init, device)]
        self.qz_ax = [DenseLayer(n_in+n_aux, n_hidden, T.relu, w_init, device), 
                      GaussianLayer(n_hidden, n_embedding, w_init, device)]
        
        # Decoder / Generative
        self.px_z  = [DenseLayer(n_embedding, n_hidden, T.relu, w_init, device), 
                      GaussianLayer(n_hidden, n_out, w_init, device)]
        self.pa_xz = [DenseLayer(n_in+n_embedding, n_hidden, T.relu, w_init, device), 
                      GaussianLayer(n_hidden, n_aux, w_init, device)]
        
        self.optimizer(self.learning_rate, self.clamping, self.qa_x + self.qz_ax + self.px_z + self.pa_xz)
        
    
    def forward(self, x, train=True):
        qa_x, qz_ax, px_z, pa_xz = self.qa_x, self.qz_ax, self.px_z, self.pa_xz
        
        n_batch = x.size()[1]
        # Inference model
        mu_qa, var_qa = qa_x[1].forward(qa_x[0].forward(x))
        qa_samples    = mu_qa + T.sqrt(var_qa) * T.rand_like(mu_qa) 
        
        ax = T.cat((x, qa_samples), dim=0)
        mu_qz, var_qz = qz_ax[1].forward(qz_ax[0].forward(ax))
        qz_samples    = mu_qz + T.sqrt(var_qz) * T.rand_like(mu_qz)
        
        # Generative model
        mu_px, var_px = px_z[1].forward(px_z[0].forward(qz_samples))
        px_samples    = mu_px + T.sqrt(var_px) * T.rand_like(mu_px)   # only to generate samples
        
        xz = T.cat((x, qz_samples), dim=0)
        mu_pa, var_pa = pa_xz[1].forward(pa_xz[0].forward(xz))
        pa_samples = mu_pa + T.sqrt(var_pa) * T.rand_like(mu_pa)
        
        loss = (gaussian_regularization(mu_qz, var_qz) + gaussian_likelihood(x, mu_px, var_px) + \
                gaussian_likelihood(qa_samples, mu_pa, var_pa) + gaussian_entropy(var_qa)) / n_batch
        
        if train:
            loss.backward()
            self.optimizer.optimize()
        return loss, px_samples
    
    
    def train(self, n_iterations, fetcher, hook):
        t0 = time()
        for j in range(n_iterations+1):
            x = fetcher()
            loss, px_samples = self.forward(x)
            # logistic
            hook(self, loss.item(), j, time() - t0, px_samples.detach().numpy())


class MMDVAE(object):
    """
    Mutual Information maximizing VAE
    """
    
    def __init__(self, params):
        self.__dict__.update(params)
        n_in, n_hidden, n_embedding, n_out = self.n_in, self.n_hidden, self.n_embedding, self.n_out
        hidden_nonlinear, w_init, device = self.hidden_nonlinear, self.w_init, self.device
        
        
        # Encoder / Inference
        self.qz_x = [DenseLayer(n_in, n_hidden, hidden_nonlinear, w_init, device), 
                     GaussianLayer(n_hidden, n_embedding, w_init, device)]
        
        # Decoder / Generative
        self.px_z = [DenseLayer(n_embedding, n_hidden, hidden_nonlinear, w_init, device), 
                     DenseLayer(n_hidden, n_in, T.sigmoid, w_init, device)]
        
        self.optimizer(self.learning_rate, self.clamping, self.qz_x + self.px_z)
    
    
    def forward(self, x, train=True):
        qz_x, px_z, alpha, scaling = self.qz_x, self.px_z, self.alpha, self.scaling
        
        n_batch = x.size()[1]
        # Inference model
        mu_z, var_z = qz_x[1].forward(qz_x[0].forward(x))
        z_prior     = T.rand_like(mu_z)
        z_samples   = mu_z + var_z * z_prior
        
        # Generative model
        x_hat = px_z[1].forward(px_z[0].forward(z_samples))
        reconstruction_loss          = bernoulli_likelihood(x, x_hat) / n_batch
        regularization_posterior     = (1.-alpha)*gaussian_regularization(mu_z, var_z) / n_batch
        regularization_ave_posterior = (alpha + scaling - 1) * -mmd(z_samples, z_prior)
        loss = reconstruction_loss + regularization_posterior + regularization_ave_posterior
        
        if train:
            loss.backward()
            self.optimizer.optimize()
        return loss, {'recon': reconstruction_loss.detach().numpy(), 
                      'reg_post': regularization_posterior.detach().numpy(), 
                      'reg_ave_post': regularization_ave_posterior.detach().numpy()}
    
    
    def train(self, n_iterations, fetcher, hook):
        t0 = time()
        for j in range(n_iterations+1):
            x = fetcher()
            loss, results = self.forward(x)
            # logistic
            hook(self, loss.item(), j, time()-t0, results                                                                                                               )