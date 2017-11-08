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

    def get_random_batch(self, batch_size):
        return [self.get_random_example() for _ in range(batch_size)]

    def get_batches(self, batch_size):
        random.shuffle(self.data)
        for i in range(0, len(self.data), batch_size):
            yield [self._unpack(example) for example in self.data[i:i + batch_size]]
