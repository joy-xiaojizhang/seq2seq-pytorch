#import matplotlib.pyplot as plt

from data import VOCAB_SIZE
from seq2seq import Seq2Seq

lr = 1e-4
batch_size = 5
epoch = 20
hidden_size = 256
print_iters = 100

def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='train')
    plt.plot(val_losses, color='blue', label='val')
    plt.legend(loc='upper right', frameon=False)
    plt.show()


model = Seq2Seq(VOCAB_SIZE, hidden_size)
train_losses, val_losses = model.train(lr, batch_size, epoch, print_iters)
plot_loss(train_losses, val_losses)
