import os, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var

from dataset import Dataset
from data import TRAIN_FILE_NAME, VAL_FILE_NAME, VOCAB_FILE_NAME, GO_TOKEN, MAX_OUTPUT_LENGTH, load_vocab, mask_seqs, indices_to_line
from nltk.translate import bleu_score as BLEU

from blackbox_gru_rnn import BlackboxGRUEncoderRNN, BlackboxGRUDecoderRNN
from mem_gru_rnn import GRUSeq2Seq

NUM_BLEU_GRAMS = 4

logs_dir = 'logs'
USE_CUDA = torch.cuda.is_available()

class Train(nn.Module):
    def __init__(self):
        super(Train, self).__init__()
        self.model = GRUSeq2Seq(1024, 512, 512, 200, 200)

    # Finds the index of given token in the input sequence
    # If the token is not in the input, return the last index
    def find_token_index(self, input_seq, token):
        try:
            idx = input_seq.index(token)
            return idx
        except ValueError:
            return len(input_seq)

    def forward(self, input_seqs, target_seqs, input_lens):
        _, encoder_hidden = self.encoder(input_seqs, input_lens)
        decoder_output, _ = self.decoder(target_seqs, encoder_hidden)
        return decoder_output

    def predict(self, input_seqs, vocab, index2word, max_output_len=MAX_OUTPUT_LENGTH):
        input_seqs = sorted(input_seqs, key=lambda p: len(p), reverse=True)
        input_lens = [len(seq) for seq in input_seqs]
        input_seqs, input_masks = mask_seqs(input_seqs)
        input_seqs = Var(torch.LongTensor(input_seqs), volatile=True)

        if USE_CUDA:
            input_seqs = input_seqs.cuda()

        batch_size = input_seqs.size()[0] 

        _, encoder_hidden = self.encoder(input_seqs, input_lens)

        decoder_hidden = encoder_hidden
        decoder_input = Var(torch.LongTensor([[vocab[GO_TOKEN]] for _ in range(batch_size)]), volatile=True)
        decoded_words = [[GO_TOKEN] for _ in range(batch_size)]

        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        for t in range(max_output_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = decoder_output.permute(1, 0, 2) # 1 x batch x vocab 
            topv, topi = decoder_output.data.topk(1) # 1 x batch x 1 
            topi = topi.permute(0, 2, 1) # 1 x 1 x batch
            word_indices = [topi[0][0][i] for i in range(batch_size)]
            words = [index2word[idx] for idx in word_indices]
            for i in range(batch_size):
                decoded_words[i].append(words[i])
            decoder_input = Var(torch.LongTensor([[idx] for idx in word_indices]))
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

        # Truncate everything after <eos>
        decoded_words = [seq[:self.find_token_index(seq, '<eos>') + 1] for seq in decoded_words]
        return decoded_words

    def get_bleu_scores(self, hypothesis, reference, num_grams=NUM_BLEU_GRAMS):
        bleu_scores = []
        for i in range(1, num_grams + 1):
            weights = tuple([1 / i for _ in range(i)])
            bleu_i_score = BLEU.sentence_bleu([reference], hypothesis, weights = weights)
            #bleu_i_score = 'BLEU-{} score: {}\n'.format(i, bleu_i_score)
            #bleu_scores += bleu_i_score
            bleu_scores.append(bleu_i_score)
        return bleu_scores

    def print_prediction_results(self, input_seqs, target_seqs, test_seqs, index2word):
        results = ''
        batch_size = input_seqs.size()[0]

        # Convert indices to words
        input_seqs = [indices_to_line(seq, index2word) for seq in input_seqs]
        target_seqs = [indices_to_line(seq, index2word) for seq in target_seqs]
        test_seqs = [indices_to_line(seq, index2word) for seq in test_seqs]

        for i in range(batch_size):
            results += '************Test Pair {}************\n'.format(i)
            results += 'Input: {}\n'.format(' '.join(input_seqs[i]))
            results += 'Expected output: {}\n'.format(' '.join(target_seqs[i]))
            results += 'Actual output: {}\n'.format(' '.join(test_seqs[i]))

            # Print bleu scores
            bleu_scores = self.get_bleu_scores(test_seqs[i], target_seqs[i])
            results += ''.join(['BLEU-{} score: {}\n'.format(i, bleu_i_score) for i, bleu_i_score in bleu_scores])

            results += '\n'
        return results, bleu_scores

    def train_mem_gru(self, lr, batch_size, epoch, print_iters, num_val_examples):
        # Experiment settings (from "A Knowledge-Grounded Neural Conversational Model")
        # hidden_size: 512
        # embed_size: 512
        # vocab_size: 1024
        # optimizer: Adam
        # lr: 0.1
        # batch_size: 128
        # clip gradient at 5
        # init weights: Normal dist in [-sqrt(d/3), sqrt(d/3)] (d=dim)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train = Dataset(TRAIN_FILE_NAME)
        val = Dataset(VAL_FILE_NAME)
        _, index2word = load_vocab(VOCAB_FILE_NAME)

        train_losses = []
        bleu_scores = []

        for e in range(epoch):
            train_batches = list(train.get_batches(batch_size))
            num_train_batches = len(train_batches)

            print('Number of training batches: {}'.format(num_train_batches))

            fo = open(os.path.join(logs_dir, 'epoch_{}.txt'.format(e)), 'w')
            epoch_start = '************Epoch {} Starts*************\n'.format(e)
            fo.write(epoch_start)
            print(epoch_start)

            epoch_start_time = time.time()
            iters_start_time = time.time()

            for i in range(1, num_train_batches):
                train_input_seqs, train_input_mask, train_max_input_len, train_target_seqs, train_target_mask = train_batches[i]

                if USE_CUDA:
                    train_input_seqs = train_input_seqs.cuda()
                    train_input_mask = train_input_mask.cuda()
                    train_target_seqs = train_target_seqs.cuda()
                    train_target_mask = train_target_mask.cuda()

                train_loss = model(train_max_input_len, train_input_seqs, train_input_mask, is_train=True, output_seqs=train_target_seqs, output_mask=train_target_mask)
                train_losses.append(train_loss)

                if i % print_iters == 0:
                    iters_end_time = time.time()

                    # Print train loss
                    string = 'Iters: {}, train loss: {:.2f}, time: {:.2f} s\n'
                    string = string.format(i, train_loss.data[0], iters_end_time - iters_start_time)
                    fo.write(string)
                    print(string)

                    # Run prediction examples
                    val_input_seqs, val_input_mask, val_max_input_len, val_target_seqs, _ = val.get_random_batch(num_val_examples)

                    if USE_CUDA:
                        val_input_seqs = val_input_seqs.cuda()
                        val_input_mask = val_input_mask.cuda()

                    predictions = model(val_max_input_len, val_input_seqs, val_input_mask, is_train=False, start_idx=1)
                    pred_results, bleu = print_prediction_results(val_input_seqs, val_target_seqs, predictions, index2word)
                    print(pred_results)
                    bleu_scores.append(bleu) #Modify later
                    iters_start_time = time.time()

            epoch_end_time = time.time()
            epoch_end = '************Epoch {} Ends*************\n'.format(e)
            epoch_time = 'Total time: {:.2f} s\n'.format(epoch_end_time - epoch_start_time)
            fo.write(epoch_end + epoch_time)
            fo.close()
            print(epoch_end + epoch_time)
            print(train_losses[-1], bleu_scores[-1])
            epoch_start_time = time.time()

        return train_losses, bleu_scores
                







