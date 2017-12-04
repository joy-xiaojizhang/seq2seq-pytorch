import os, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var

from dataset import Dataset
from data import TRAIN_FILE_NAME, VAL_FILE_NAME, VOCAB_FILE_NAME, VOCAB_SIZE, MAX_OUTPUT_LENGTH, load_vocab, mask_seqs, indices_to_line, GO_TOKEN_INDEX, EOS_TOKEN_INDEX
from nltk.translate import bleu_score as BLEU

from blackbox_gru_rnn import BlackboxGRUEncoderRNN, BlackboxGRUDecoderRNN
from mem_gru_rnn import GRUSeq2Seq

NUM_BLEU_GRAMS = 4

logs_dir = 'logs'
USE_CUDA = torch.cuda.is_available()

class Train(nn.Module):
    def __init__(self):
        super(Train, self).__init__()
        self.model = GRUSeq2Seq(VOCAB_SIZE, 300, 300, 100, 100)

    def get_bleu_scores(self, hypothesis, reference, num_grams=NUM_BLEU_GRAMS):
        bleu_scores = []
        for i in range(1, num_grams + 1):
            weights = tuple([1 / i for _ in range(i)])
            bleu_i_score = BLEU.sentence_bleu([reference], hypothesis, weights = weights)
            bleu_scores.append(bleu_i_score)
        return bleu_scores

    def truncate_seq(self, seq, token):
        # Find token index in seq
        try:
            idx = seq.index(token)
        except ValueError:
            idx = len(seq)
        return seq[:idx + 1]

    def print_prediction_results(self, input_seqs, target_seqs, pred_seqs, index2word, eos_idx):
        results = ''
        batch_size = input_seqs.size()[0]

        # Truncate sequences (remove everything after <eos>)
        input_seqs = [self.truncate_seq(seq.data.cpu().numpy().tolist(), eos_idx) for seq in input_seqs] 
        target_seqs = [self.truncate_seq(seq.data.cpu().numpy().tolist(), eos_idx) for seq in target_seqs] 
        pred_seqs = [self.truncate_seq(seq.data.cpu().numpy().tolist(), eos_idx) for seq in pred_seqs] 

        # Convert indices to words
        input_seqs = [indices_to_line(seq, index2word) for seq in input_seqs]
        target_seqs = [indices_to_line(seq, index2word) for seq in target_seqs]
        pred_seqs = [indices_to_line(seq, index2word) for seq in pred_seqs]

        for i in range(batch_size):
            results += '************Test Pair {}************\n'.format(i)
            results += 'Input: {}\n'.format(' '.join(input_seqs[i]))
            results += 'Expected output: {}\n'.format(' '.join(target_seqs[i]))
            results += 'Actual output: {}\n'.format(' '.join(pred_seqs[i]))

            # Print bleu scores
            bleu_scores = self.get_bleu_scores(pred_seqs[i], target_seqs[i])
            results += ''.join(['BLEU-{} score: {}\n'.format(i + 1, bleu_i_score) for i, bleu_i_score in enumerate(bleu_scores)])

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
                optimizer.zero_grad()

                train_input_seqs, train_input_mask, train_max_input_len, train_target_seqs, train_target_mask, train_max_target_len = train.prepare_batched_input(train_batches[i], USE_CUDA, False)

                train_loss = self.model(train_max_input_len, train_input_seqs, train_input_mask, train_max_target_len, is_train=True, output_seqs=train_target_seqs, output_mask=train_target_mask)
                train_loss.backward()
                train_losses.append(train_loss)
                optimizer.step()

                if i % print_iters == 0:
                    iters_end_time = time.time()

                    # Print train loss
                    string = 'Iters: {}, train loss: {:.2f}, time: {:.2f} s\n'
                    string = string.format(i, train_loss.data[0], iters_end_time - iters_start_time)
                    fo.write(string)
                    print(string)

                    # Run prediction examples
                    # Set volatile to True for inference mode
                    val_input_seqs, val_input_mask, val_max_input_len, val_target_seqs, _, _ = val.get_random_batch(num_val_examples, USE_CUDA, True)

                    predictions = self.model(val_max_input_len, val_input_seqs, val_input_mask, 25, is_train=False, start_idx=GO_TOKEN_INDEX)
                    pred_results, bleu = self.print_prediction_results(val_input_seqs, val_target_seqs, predictions, index2word, EOS_TOKEN_INDEX)
                    fo.write(pred_results)
                    print(pred_results)
                    bleu_scores.append(bleu) #Modify later
                    iters_start_time = time.time()

            epoch_end_time = time.time()
            epoch_end = '************Epoch {} Ends*************\n'.format(e)
            epoch_time = 'Total time: {:.2f} s\n'.format(epoch_end_time - epoch_start_time)
            fo.write(epoch_end + epoch_time)
            fo.write(train_losses[-1], bleu_scores[-1])
            fo.close()
            print(epoch_end + epoch_time)
            epoch_start_time = time.time()

        return train_losses, bleu_scores
