import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var

'''
A gated recurrent unit cell
Math:
r = \mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr})
z = \mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz})
n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn}))
h' = (1 - z) * n + z * h
'''
def GRUcell(nn.Module):
    def __init__(self, hidden_size, context_size, num_slots):
        # Define size parameters
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_slots = num_slots
        self.input_size = hidden_size + context_size

        # Define GRU unit parameters
        # W: learnable input-hidden weights (input x (3*hidden)): W_ir | W_iz | W_in
        # U: learnable hidden-hidden weights (input x (3*hidden)): W_hr | W_hz | W_hn
        # bi: learnable input-hidden bias (1 x (3*hidden)): b_ir | b_iz | b_in
        # bh: learnable hidden-hidden bias (1 x (3*hidden)): b_hr | b_hz | b_hn
        # sigma_g, sigma_h: activation functions
        self.W = Var(torch.zeros(self.input_size, 3 * hidden_size))
        self.U = Var(torch.zeros(hidden_size, 3 * hidden_size)) 
        self.bi = Var(torch.zeros(3 * hidden_size))
        self.bh = Var(torch.zeros(3 * hidden_size))
        self.sigma_g = torch.nn.sigmoid()
        self.sigma_h = torch.nn.tanh()

        # Define memory units
        self.Mk = Var(torch.zeros(num_slots, hidden_size))
        self.Mv = Var(torch.zeros(num_slots, context_size))

    def forward(self, embedded, mask, last_hidden):
        # embedded:    batch x hidden
        # mask:        batch x hidden
        # last_hidden: batch x hidden

        # Compute attention and normalize
        alpha_tilde = torch.exp(torch.mm(last_hidden, Mk.transpose(0, 1))) # batch x num_slots
        alpha = alpha_tilde / torch.sum(alpha_tilde, dim = 2) # batch x num_slots

        # Compute context vector
        context = torch.mm(alpha, Mv) # batch x context

        # Compute input
        input = torch.cat((embedded, context), dim = 1) # batch x input_size

        # Compute gates
        # r (reset gate): batch x hidden
        # z (update gate): batch x hidden
        # n (output gate): batch x hidden
        split = 2 * hidden_size
        gates = self.sigma_g(torch.mm(input, self.W[:, :split]) +
                             torch.mm(last_hidden, self.U[:, :split]) +
                             self.bi[:split] + self.bh[:split]) # batch x (2*hidden)
        r = gates[:, :self.hidden_size]
        z = gates[:, self.hidden_size:] 
        n = self.sigma_h(torch.mm(input, self.W[:, split:]) + self.bi[split:] +
                         r * torch.mm(last_hidden, self.U[:, split:]) + self.bh[split:])

        # Compute hidden state
        ones_batch_hidden = Var(torch.ones(self.batch_size, self_hidden_size))
        h_tilde = (ones_batch_hidden - z) * n + z * last_hidden # batch x hidden
        h = mask * h_tilde + (ones_batch_hidden - mask) * last_hidden # batch x hidden

        return h


def RNNStep():
    def __init__(self):
        pass


def GRUEncoderRNN(nn.Module):
    def __init__(self):
        super(GRUEncoderRNN, self).__init__()
        self.rnn = RNNStep()

    def forward(self, input_seqs):
        pass


def GRUDecoderRNN(nn.Module):
    def __init__(self):
        super(GRUDecoderRNN, self).__init__()
        self.rnn = RNNStep()

    def forward(self, input_seqs):
        pass
