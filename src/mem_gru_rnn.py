import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var

import numpy as np
import load_data

USE_CUDA = torch.cuda.is_available()

'''
A gated recurrent unit cell with memory pad
Memory pad:
a' = exp(h * M_k)
a = normalize(a')
c = a' * M_v
x = (in, c)

GPU:
r = \mathrm{sigmoid}(W_i_{ir} x + b_{ir} + W_i_{hr} h + b_{hr})
z = \mathrm{sigmoid}(W_i_{iz} x + b_{iz} + W_i_{hz} h + b_{hz})
n = \tanh(W_i_{in} x + b_{in} + r * (W_i_{hn} h + b_{hn}))
h' = (1 - z) * n + z * h
'''
class GRUcell(nn.Module):
    # TODO: Read papers & code on how to choose init weights 
    @staticmethod
    def init_weights(dims):
        w = (np.random.randn(*dims) * 0.08)
        w = torch.nn.Parameter(torch.from_numpy(w.astype('float32')), requires_grad=True)
        if USE_CUDA:
            w = w.cuda()
        return w

    def __init__(self, embed_size, hidden_size, context_size, num_slots):
        super(GRUcell, self).__init__()

        # Define size parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_slots = num_slots
        self.input_size = embed_size + context_size

        # Define GRU unit parameters
        # W_i: learnable input-hidden weights (input x (3*hidden)): W_ir | W_iz | W_in
        # W_h: learnable hidden-hidden weights (input x (3*hidden)): W_hr | W_hz | W_hn
        # b_i: learnable input-hidden bias (1 x (3*hidden)): b_ir | b_iz | b_in
        # b_h: learnable hidden-hidden bias (1 x (3*hidden)): b_hr | b_hz | b_hn
        # sigma_g, sigma_h: activation functions
        self.W_i = self.init_weights((self.input_size, 3 * hidden_size))
        self.W_h = self.init_weights((hidden_size, 3 * hidden_size))
        self.b_i = self.init_weights((1, 3 * hidden_size))
        self.b_h = self.init_weights((1, 3 * hidden_size))
        self.sigma_g = nn.Sigmoid()
        self.sigma_h = nn.Tanh()

        # Define memory units
        self.M_k = self.init_weights((num_slots, hidden_size))
        self.M_v = self.init_weights((num_slots, context_size))

    def forward(self, embedded, mask, last_hidden = None):
        # embedded:    batch x hidden
        # mask:        batch x 1 
        # last_hidden: batch x hidden

        batch_size = embedded.size()[0]

        # Initialize first hidden state
        if last_hidden is None:
            split = 2 * self.hidden_size
            context = Var(torch.zeros(batch_size, self.context_size)) # batch x context
            if USE_CUDA:
                context = context.cuda()
            input = torch.cat((embedded, context), dim = 1) # batch x input_size
            return torch.mm(input, self.W_i[:, split:])
            # Or simply:
            # return torch.zeros(batch_size, self.hidden_size)

        # Compute attention and normalize
        alpha_tilde = torch.exp(torch.mm(last_hidden, self.M_k.transpose(0, 1))) # batch x num_slots
        alpha_sum = torch.sum(alpha_tilde, dim = 1).view(-1, 1) # batch x 1
        alpha = torch.div(alpha_tilde, alpha_sum.expand(alpha_tilde.size())) # batch x num_slots

        # Compute context vector
        context = torch.mm(alpha, self.M_v) # batch x context
        if USE_CUDA:
            context = context.cuda()

        # Compute input
        input = torch.cat((embedded, context), dim = 1) # batch x input_size

        # Compute gates
        # r (reset gate): batch x hidden
        # z (update gate): batch x hidden
        # n (output gate): batch x hidden
        split = 2 * self.hidden_size
        b_i = self.b_i.repeat(batch_size, 1) # batch x (3*hidden)
        b_h = self.b_h.repeat(batch_size, 1) # batch x (3*hidden)
        gates = self.sigma_g(torch.mm(input, self.W_i[:, :split]) +
                             torch.mm(last_hidden, self.W_h[:, :split]) +
                             b_i[:, :split] + b_h[:, :split]) # batch x (2*hidden)
        r = gates[:, :self.hidden_size]
        z = gates[:, self.hidden_size:] 
        n = self.sigma_h(torch.mm(input, self.W_i[:, split:]) + b_i[:, split:] +
                         r * torch.mm(last_hidden, self.W_h[:, split:]) + b_h[:, split:])

        # Compute hidden state
        ones_batch_hidden = Var(torch.ones(batch_size, self.hidden_size))

        if USE_CUDA:
            ones_batch_hidden = ones_batch_hidden.cuda()

        h = (ones_batch_hidden - z) * n + z * last_hidden # batch x hidden

        # Masking
        mask = mask.repeat(self.hidden_size, 1).transpose(0, 1) # batch x hidden
        h = mask * h + (ones_batch_hidden - mask) * last_hidden # batch x hidden

        return h


class GRUEncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, context_size, num_slots, prev_embed = None):
        super(GRUEncoderRNN, self).__init__()

        # Define size parameters
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_slots = num_slots

        # Define embedding
        if prev_embed:
            self.embedding = prev_embed
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        # Define GRU
        self.gru_cell = GRUcell(embed_size, hidden_size, context_size, num_slots) 

    def forward(self, max_step, input_seqs, input_mask):
        # max_step: maximum number of time steps (max length of input seqs)
        # input_seqs: batch x max_steps (index representation)
        # input_mask: batch x max_steps (binary, float representation)

        embedded = self.embedding(input_seqs) # batch x max_step x embed_size
        hidden = None
        for i in range(max_step):
            embedded_i = embedded[:, i, :].squeeze() # batch x embed_size
            mask_i = input_mask[:, i] # batch
            hidden = self.gru_cell(embedded_i, mask_i, hidden) # batch x hidden
        return hidden


class GRUDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, context_size, num_slots, prev_embed = None):
        super(GRUDecoderRNN, self).__init__()

        # Define size parameters
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_slots = num_slots

        # Define embedding
        if prev_embed:
            self.embedding = prev_embed
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        # Define network parameters
        self.gru_cell = GRUcell(embed_size, hidden_size, context_size, num_slots) 
        self.out = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(size_average = False, ignore_index = 2)


    def forward(self, max_step, hidden, is_train = True, output_seqs = None, output_mask = None, start_idx = None):
        # hidden: batch x hidden
        # output_seqs: batch x max_step (index representation), always starts with GO_TOKEN
        # output_mask: batch x max_step (binary, float representation)
        # Train mode: use output_seqs as input, return training loss
        # Predict mode: use GO_TOKEN as input, return predicted seqs
        
        # Train mode
        if is_train:
            loss = 0.0
            embedded = self.embedding(output_seqs) # batch x max_step x embed_size

            for i in range(max_step - 1): # don't feed <eos>, might change later
                embedded_i = embedded[:, i, :].squeeze() # batch x embed_size
                mask_i = output_mask[:, i] # batch
                hidden = self.gru_cell(embedded_i, mask_i, hidden) # batch x hidden
                logit = self.out(hidden) # batch x vocab
                loss_i = self.loss_fn(logit, output_seqs[:, i + 1]) 
                loss += loss_i

            return loss / torch.sum(output_mask)
        
        # Predict mode
        batch_size = hidden.size()[0]
        mask_ones = Var(torch.ones([batch_size])) # batch x 1
        decoder_input = Var(torch.LongTensor([[start_idx] for _ in range(batch_size)])) # batch x 1
        pred_seqs = decoder_input.clone() 

        if USE_CUDA:
            mask_ones = mask_ones.cuda()
            decoder_input = decoder_input.cuda()
            pred_seqs = pred_seqs.cuda()

        for i in range(max_step - 1): # May modify later
            embedded_i = self.embedding(decoder_input).squeeze() # batch x embed
            hidden = self.gru_cell(embedded_i, mask_ones, hidden) # batch x hidden
            logit = self.out(hidden) # batch x vocab

            # Get argmax of each batch and assign as new input
            _, pred = torch.max(logit, 1) # batch
            pred = pred.view((-1, 1)) # batch x 1
            decoder_input = pred
 
            pred_seqs = torch.cat((pred_seqs, pred), dim = 1)
            
        return pred_seqs


class GRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, context_size, num_slots, prev_embed = None):
        super(GRUSeq2Seq, self).__init__()
        self.encoder = GRUEncoderRNN(vocab_size, embed_size, hidden_size, context_size, num_slots, prev_embed)
        self.decoder = GRUDecoderRNN(vocab_size, embed_size, hidden_size, context_size, num_slots, prev_embed)

    def forward(self, input_max_step, input_seqs, input_mask, output_max_step, is_train = True, output_seqs = None, output_mask = None, start_idx = None):
        # Encoding step
        encoder_hidden = self.encoder(input_max_step, input_seqs, input_mask)

        # Decoding step: train mode
        if is_train:
            loss = self.decoder(output_max_step, encoder_hidden, is_train = True, output_seqs = output_seqs, output_mask = output_mask)
            return loss

        # Decoding step: predict mode
        predictions = self.decoder(output_max_step, encoder_hidden, is_train = False, start_idx = start_idx)
        return predictions
