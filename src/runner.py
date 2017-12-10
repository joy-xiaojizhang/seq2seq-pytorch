#import matplotlib.pyplot as plt
import sys
from train import Train
from data import VOCAB_SIZE

train_lr = 1e-4
train_batch_size = 50
train_epoch = 20
train_hidden_size = 300
train_print_iters = 100
num_val_examples = 10

def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='train')
    plt.plot(val_losses, color='blue', label='val')
    plt.legend(loc='upper right', frameon=False)
    plt.show()

model = Train()
train_losses, bleu_scores = model.train_mem_gru(train_lr, train_batch_size, train_epoch, train_print_iters, num_val_examples)
plot_loss(train_losses, val_losses)
