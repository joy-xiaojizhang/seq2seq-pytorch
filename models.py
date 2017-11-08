import os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable as Var

from dataset import Dataset
from data import TRAIN_FILE_NAME, VAL_FILE_NAME, pad_seqs
from nltk.translate import bleu_score as BLEU

num_bleu_grams = 4

logs_dir = 'logs'
use_cuda = torch.cuda.is_available()

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = DecoderRNN(vocab_size, hidden_size)
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def _get_loss(self, batch, loss_fn):
        answer_lens = [len(example['answer']) for example in batch]
        questions = pad_seqs([example['question'] for example in batch])
        answers = pad_seqs([example['answer'] for example in batch])

        questions = Var(torch.LongTensor(questions))
        answers = Var(torch.LongTensor(answers))

        if use_cuda:
            questions = questions.cuda()
            answers = answers.cuda()

        output = self(questions, answers)

        loss = 0
        batch_size = len(batch)
        for i in range(batch_size):
            loss += loss_fn(output[i, :answer_lens[i] - 1], answers[i, 1:answer_lens[i]])

        return loss / batch_size

    def forward(self, input_seqs, target_seqs):
        _, encoder_hidden = self.encoder(input_seqs)
        decoder_output, _ = self.decoder(target_seqs, encoder_hidden)
        return decoder_output

    def get_bleu_score(self, num_grams):
        bleu_scores = ''
        for i in range(1, num_grams + 1):
            weights = tuple([1 / n for _ in range(n)])
            BLEUscore = BLEU.sentence_bleu([reference], hypothesis, weights = weights)
            bleu_i_score = 'BLEU-{} score: {}\n'.format(i, BLEUscore)
            print(bleu_i_score)
            bleu_scores += bleu_i_score
        return bleu_scores

    def train(self, lr, batch_size, epoch, print_iters):
        optimizer = SGD(self.parameters(), lr=lr)
        encoder_hidden = self.encoder.init_hidden(batch_size)

        train_losses = []
        val_losses = []

        train = Dataset(TRAIN_FILE_NAME)
        val = Dataset(VAL_FILE_NAME)

        for e in range(epoch):
            # Shuffle data at the beginning of every epoch
            train_batches = list(train.get_batches(batch_size))
            num_train_batches = len(train_batches)
            print("Number of training batches: {}".format(num_train_batches))

            fo = open(os.path.join(logs_dir, 'epoch_{}.txt'.format(e)), 'w+')
            epoch_start = "************Epoch {} Starts*************\n".format(e)
            fo.write(epoch_start)
            print(epoch_start)

            epoch_start_time = time.time()
            iters_start_time = time.time()

            for i in range(1, num_train_batches):
                #train_batch = train.get_random_batch(batch_size)
                train_batch = train_batches[i]
                val_batch = val.get_random_batch(batch_size)
                loss_fn = torch.nn.NLLLoss()

                train_loss = self._get_loss(train_batch, loss_fn)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                val_loss = self._get_loss(val_batch, loss_fn)

                train_losses.append(train_loss.data[0])
                val_losses.append(val_loss.data[0])

                if i % print_iters == 0:
                    iters_end_time = time.time()
                    string = 'Iters: {}, train loss: {:.2f}, val loss: {:.2f}, time: {:.2f} s\n'
                    string = string.format(i, train_loss.data[0], val_loss.data[0], iters_end_time - iters_start_time)
                    #string += get_bleu_score(num_bleu_grams)
                    fo.write(string)
                    print(string)
                    iters_start_time = time.time()

            epoch_end_time = time.time()
            epoch_end = "************Epoch {} Ends*************\n".format(e)
            epoch_time = "Total time: {:.2f} s\n".format(epoch_end_time - epoch_start_time)
            fo.write(epoch_end + epoch_time)
            print(epoch_end + epoch_time)
            epoch_start_time = time.time()

        return train_losses, val_losses

    def evaluate(self, input_seq):
        pass


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, batch_first=True)

    def init_hidden(self, batch_size):
        result = Var(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        batch_size = embedded.size()[0]
        output = embedded
        hidden = self.init_hidden(batch_size)
        for _ in range(self.num_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()

    def init_hidden(self, batch_size):
        result = Var(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            result = result.cuda()
        return result

    @staticmethod
    def create_rnn_input(embedded, thought):
        # reorder axes to be (seq_len, batch_size, hidden_size)
        embedded = embedded.permute(1, 0, 2)
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
        if use_cuda:
            result = result.cuda()
        batch_size = linear_output.size()[0]
        for i in range(batch_size):
            result[i] = self.softmax(linear_output[i])
        return result

    def forward(self, target_seqs, thought):
        target_seqs = self.embedding(target_seqs)
        rnn_input = self.create_rnn_input(target_seqs, thought)
        batch_size = target_seqs.size()[0]
        output = rnn_input
        hidden = self.init_hidden(batch_size)
        for _ in range(self.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax_batch(self.out(output))
        return output, hidden
