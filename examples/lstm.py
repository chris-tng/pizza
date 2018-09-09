import sys
sys.path.append('../..')

from pizza.models import LSTMLayer, DenseLayer
import torch as T


class LSTMModel(object):
    """
    2-layer LSTM model
    """
    
    def __init__(self, params):
        self.__dict__.update(params)
        device, w_init = self.device, self.w_init
        self.lstm1  = LSTMLayer(self.n_in, self.n_hidden1, w_init=w_init, device=device)
        self.lstm2  = LSTMLayer(self.n_hidden1, self.n_hidden2, w_init=w_init, device=device)
        self.output = DenseLayer(self.n_hidden2, self.n_out, self.out_nonlinear, w_init=w_init, device=device)
        self.optimizer(self.learning_rate, self.clamping, [self.lstm1, self.lstm2, self.output])
       
    
    def forward(self, x, y, training=True):
        """
        Perform forward-backward pass
        
        Deactive backward pass if `training` is False, useful for prediction
        """
        loss = 0.
        n_batch = x.size()[1]
        hs1     = self.lstm1.forward(x)
        hs2     = self.lstm2.forward(hs1) 
        y_preds = []
        for j in range(n_batch):
            y_pred = self.output.forward(hs2[:, j].view(-1, 1))
            loss  += self.compute_loss(y_pred, y[:, j])
            y_preds.append(y_pred.detach().numpy())    # for logistics
            
        loss /= n_batch
        if training:
            loss.backward()
            self.optimizer.optimize()
            
        return loss, y_preds
        
        
    def train(self, n_iterations, fetcher, hook):
        for j in range(n_iterations+1):
            x, y = fetcher()
            loss, y_preds = self.forward(x, y)
            # logistic
            hook(self, loss.item(), y_preds, j, x.detach().cpu().numpy(), y.detach().cpu().numpy())

