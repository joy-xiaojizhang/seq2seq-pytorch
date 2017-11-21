import os, time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from dataset import Dataset
from data import TRAIN_FILE_NAME, VAL_FILE_NAME, VOCAB_FILE_NAME, GO_TOKEN, MAX_OUTPUT_LENGTH, load_vocab, mask_seqs, indices_to_line
from nltk.translate import bleu_score as BLEU

NUM_BLEU_GRAMS = 4

logs_dir = 'logs'
use_cuda = torch.cuda.is_available()

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = VanillaDecoderRNN(vocab_size, hidden_size)
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def _get_loss(self, batch, loss_fn):
        input_seqs = [example['question'] for example in batch]
        target_seqs = [example['answer'] for example in batch]

        # Zip into pairs, sort by length (descending), unzip 
        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)

        input_lens = [len(seq) for seq in input_seqs]
        target_lens = [len(seq) for seq in target_seqs]
 
        input_seqs, input_masks = mask_seqs(input_seqs)
        target_seqs, target_masks = mask_seqs(target_seqs)

        input_seqs = Var(torch.LongTensor(input_seqs))
        target_seqs = Var(torch.LongTensor(target_seqs))

        if use_cuda:
            input_seqs = input_seqs.cuda()
            target_seqs = target_seqs.cuda()

        output = self(input_seqs, target_seqs, input_lens)

        loss = 0
        batch_size = len(batch)
        for i in range(batch_size):
            loss += loss_fn(output[i, :target_lens[i] - 1], target_seqs[i, 1:target_lens[i]])

        return loss / batch_size

    def forward(self, input_seqs, target_seqs, input_lens):
        _, encoder_hidden = self.encoder(input_seqs, input_lens)
        decoder_output, _ = self.decoder(target_seqs, encoder_hidden)
        return decoder_output

    def evaluate(self, input_seqs, vocab, index2word, max_output_len=MAX_OUTPUT_LENGTH):
        input_seqs = sorted(input_seqs, key=lambda p: len(p), reverse=True)
        input_lens = [len(seq) for seq in input_seqs]
        input_seqs, input_masks = mask_seqs(input_seqs)
        input_seqs = Var(torch.LongTensor(input_seqs), volatile=True)

        if use_cuda:
            input_seqs = input_seqs.cuda()

        batch_size = input_seqs.size()[0] 

        _, encoder_hidden = self.encoder(input_seqs, input_lens)

        decoder_hidden = encoder_hidden
        decoder_input = Var(torch.LongTensor([[vocab[GO_TOKEN]] for _ in range(batch_size)]), volatile=True)
        decoded_words = [[GO_TOKEN] for _ in range(batch_size)]

        if use_cuda:
            decoder_input = decoder_input.cuda()

        for t in range(max_output_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = decoder_output.permute(1, 0, 2)
            topv, topi = decoder_output.data.topk(1)
            topi = topi.permute(0, 2, 1) # 1 x 1 x batch_size
            word_indices = [topi[0][0][i] for i in range(batch_size)]
            words = [index2word[idx] for idx in word_indices]
            for i in range(batch_size):
                decoded_words[i].append(words[i])
            decoder_input = Var(torch.LongTensor([[idx] for idx in word_indices]))
            if use_cuda:
                decoder_input = decoder_input.cuda()
        decoded_words = [seq[:seq.index('<eos>') + 1] for seq in decoded_words]
        return decoded_words

    def get_bleu_score(self, hypothesis, reference, num_grams=NUM_BLEU_GRAMS):
        bleu_scores = ''
        for i in range(1, num_grams + 1):
            weights = tuple([1 / i for _ in range(i)])
            bleu_i_score = BLEU.sentence_bleu([reference], hypothesis, weights = weights)
            bleu_i_score = 'BLEU-{} score: {}\n'.format(i, bleu_i_score)
            bleu_scores += bleu_i_score
        return bleu_scores

    def print_evaluation_results(self, input_seqs, target_seqs, test_seqs):
        results = ''
        batch_size = len(input_seqs)
        print('Input seqs: {}'.format(input_seqs))
        print('Test seqs: {}'.format(test_seqs))
        print('Target seqs: {}'.format(target_seqs))
        for i in range(batch_size):
            results += '************Test Pair {}************\n'.format(i)
            results += 'Input: {}\n'.format(' '.join(input_seqs[i]))
            results += 'Expected output: {}\n'.format(' '.join(target_seqs[i]))
            results += 'Actual output: {}\n'.format(' '.join(test_seqs[i]))
            results += self.get_bleu_score(test_seqs[i], target_seqs[i])
            results += '\n'
        return results

    def train(self, lr, batch_size, epoch, print_iters):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        encoder_hidden = self.encoder.init_hidden(batch_size)

        train_losses = []
        val_losses = []

        train = Dataset(TRAIN_FILE_NAME)
        val = Dataset(VAL_FILE_NAME)

        vocab, index2word = load_vocab(VOCAB_FILE_NAME)

        for e in range(epoch):
            # Shuffle data at the beginning of every epoch
            train_batches = list(train.get_batches(batch_size))
            num_train_batches = len(train_batches)
            print("Number of training batches: {}".format(num_train_batches))

            fo = open(os.path.join(logs_dir, 'epoch_{}.txt'.format(e)), 'w')
            epoch_start = "************Epoch {} Starts*************\n".format(e)
            fo.write(epoch_start)
            print(epoch_start)

            epoch_start_time = time.time()
            iters_start_time = time.time()

            for i in range(1, num_train_batches):
                #train_batch = train.get_random_batch(batch_size)
                train_batch = train_batches[i]
                val_batch = val.get_random_batch(2)
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

                    # Validation
                    val_input_seqs = [example['question'] for example in val_batch]
                    val_target_seqs = [example['answer'] for example in val_batch]

                    decoded_words = self.evaluate(val_input_seqs, vocab, index2word)

                    val_input_seqs = [indices_to_line(indices, index2word) for indices in val_input_seqs]
                    val_target_seqs = [indices_to_line(indices, index2word) for indices in val_target_seqs]
                    eval_results = self.print_evaluation_results(val_input_seqs, val_target_seqs, decoded_words)
                    fo.write(eval_results)
                    print(eval_results)
                    iters_start_time = time.time()

            epoch_end_time = time.time()
            epoch_end = "************Epoch {} Ends*************\n".format(e)
            epoch_time = "Total time: {:.2f} s\n".format(epoch_end_time - epoch_start_time)
            fo.write(epoch_end + epoch_time)
            print(epoch_end + epoch_time)
            epoch_start_time = time.time()

        return train_losses, val_losses


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

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs).transpose(0, 1) # max_len x batch x hidden_size
        packed = pack_padded_sequence(embedded, input_lengths)
        batch_size = embedded.size()[0]
        #hidden = self.init_hidden(batch_size) # num_layers x batch x hidden_size
        output, hidden = self.gru(packed, hidden)
        output, output_lengths = pad_packed_sequence(output) # unpack (back to padded)
        #print("hidden size: {}".format(hidden.size()))
        return output, hidden


class VanillaDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(VanillaDecoderRNN, self).__init__()
        self.vocab_size = vocab_size
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
