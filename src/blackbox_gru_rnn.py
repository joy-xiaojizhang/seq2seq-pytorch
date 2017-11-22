import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BlackboxGRUEncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(BlackboxGRUEncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, batch_first=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs).transpose(0, 1) # max_len x batch x hidden
        packed = pack_padded_sequence(embedded, input_lengths)
        batch_size = embedded.size()[0]
        output, hidden = self.gru(packed, hidden)
        # output size: max_len x batch x hidden
        # hidden size: num_layers x batch x hidden

        output, output_lengths = pad_packed_sequence(output)
        return output, hidden


class BlackboxGRUDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, use_cuda=False, num_layers=1):
        super(BlackboxGRUDecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()

    def init_hidden(self, batch_size):
        result = Var(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if self.use_cuda:
            result = result.cuda()
        return result

    @staticmethod
    def create_rnn_input(embedded, thought, use_cuda):
        embedded = embedded.permute(1, 0, 2) # seq_len x batch x hidden
        seq_len, batch_size, hidden_size = embedded.size()
        rnn_input = Var(torch.zeros((seq_len, batch_size, 2 * hidden_size)))
        if use_cuda:
            rnn_input = rnn_input.cuda()
        for i in range(seq_len):
            for j in range(batch_size):
                rnn_input[i, j] = torch.cat((embedded[i, j], thought[0, j]))
        # make batch first
        return rnn_input.permute(1, 0, 2)

    def softmax_batch(self, linear_output):
        result = Var(torch.zeros(linear_output.size()))
        if self.use_cuda:
            result = result.cuda()
        batch_size = linear_output.size()[0]
        for i in range(batch_size):
            result[i] = self.softmax(linear_output[i])
        return result

    def forward(self, target_seqs, thought):
        target_seqs = self.embedding(target_seqs) # batch x seq_len x hidden
        rnn_input = self.create_rnn_input(target_seqs, thought, self.use_cuda)
        batch_size = target_seqs.size()[0]
        output = rnn_input # batch x seq_len x (2 * hidden)
        hidden = self.init_hidden(batch_size) # num_layers x batch x hidden
        for _ in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax_batch(self.out(output)) # batch x seq_len x vocab 
        return output, hidden
