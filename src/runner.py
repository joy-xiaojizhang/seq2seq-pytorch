#import matplotlib.pyplot as plt
import sys
from data import VOCAB_SIZE
from seq2seq import Seq2Seq

train_lr = 1e-4
train_batch_size = 20
train_epoch = 20
train_hidden_size = 300
train_print_iters = 100

test_lr = 1e-6
test_batch_size = 3
test_epoch = 100
test_hidden_size = 500
test_print_iters = 1

def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='train')
    plt.plot(val_losses, color='blue', label='val')
    plt.legend(loc='upper right', frameon=False)
    plt.show()


if len(sys.argv) == 2 and sys.argv[1] == 'test':
    model = Seq2Seq(VOCAB_SIZE, test_hidden_size)
    train_losses, val_losses = model.train(test_lr, test_batch_size, test_epoch, test_print_iters, test_mode=True)
    plot_loss(train_losses, val_losses)
else:
    model = Seq2Seq(VOCAB_SIZE, train_hidden_size)
    train_losses, val_losses = model.train(train_lr, train_batch_size, train_epoch, train_print_iters)
    plot_loss(train_losses, val_losses)
