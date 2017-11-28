import random
import torch

from data import get_seqs, mask_seqs
from torch.autograd import Variable as Var

class Dataset(object):
    def __init__(self, file_name):
        self.data = get_seqs(file_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self._unpack(self.data[item])

    @staticmethod
    def _unpack(example):
        question, answer = example
        return {'question': question, 'answer': answer}

    def get_random_example(self):
        return self._unpack(random.choice(self.data))

    def get_random_batch(self, batch_size, use_cuda):
        return self.prepare_batched_input([self.get_random_example() for _ in range(batch_size)], use_cuda)

    def get_batches(self, batch_size):
        random.shuffle(self.data)
        for i in range(0, len(self.data), batch_size):
            yield [self._unpack(example) for example in self.data[i:i + batch_size]]

    def prepare_batched_input(self, batch, use_cuda):
        input_seqs = [example['question'] for example in batch]
        target_seqs = [example['answer'] for example in batch]
        max_input_len = max([len(seq) for seq in input_seqs])
        max_target_len = max([len(seq) for seq in target_seqs])

        input_seqs, input_mask = mask_seqs(input_seqs)
        target_seqs, target_mask = mask_seqs(target_seqs)

        input_seqs = Var(torch.LongTensor(input_seqs))
        input_mask = Var(torch.FloatTensor(input_mask))
        target_seqs = Var(torch.LongTensor(target_seqs))
        target_mask = Var(torch.FloatTensor(target_mask))

        if use_cuda:
            input_seqs = input_seqs.cuda()
            input_mask = input_mask.cuda()
            target_seqs = target_seqs.cuda()
            target_mask = target_mask.cuda()

        return input_seqs, input_mask, max_input_len, target_seqs, target_mask, max_target_len
