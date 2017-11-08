import random

from data import get_seqs

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
