import ast
import os
import random
from collections import Counter
from itertools import chain

from nltk import word_tokenize
import numpy as np

DATA_PATH = 'data'
CORPUS_PATH = os.path.join(DATA_PATH, 'cornell movie-dialogs corpus')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')

VOCAB_FILE_NAME = 'vocab.txt'

TRAIN_FILE_NAME = 'train.txt'
VAL_FILE_NAME = 'val.txt'
TEST_FILE_NAME = 'test.txt'
SAMPLE_FILE_NAME = 'sample.txt'

CORPUS_ENCODING = 'ISO-8859-2'

EOS_TOKEN = '<eos>'
GO_TOKEN = '<go>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

VOCAB_SIZE = 20000
MAX_OUTPUT_LENGTH = 25


def get_raw_data():
    data = []
    for c in get_conversations():
        num_movie_lines = len(c)
        for i in range(num_movie_lines - 1):
            qa_pair = (c[i], c[i + 1])
            data.append(qa_pair)
    return data


def get_conversations():
    conversations_path = os.path.join(CORPUS_PATH, 'movie_conversations.txt')
    conversations = load_raw_data(conversations_path)
    id_to_line = get_id_to_line()
    for c in conversations:
        line_id_list = c[-1]  # represented as a string
        yield [id_to_line[line_id] for line_id in ast.literal_eval(line_id_list)]


def get_id_to_line():
    lines_path = os.path.join(CORPUS_PATH, 'movie_lines.txt')
    lines = load_raw_data(lines_path)
    id_to_line = {}
    for l in lines:
        line_id = l[0]
        line_text = l[-1]
        id_to_line[line_id] = line_text
    return id_to_line


def load_raw_data(file_path):
    seperator = ' +++$+++ '
    with open(file_path, 'r', encoding = CORPUS_ENCODING) as f:
        for line in f:
            yield line.strip().split(seperator)


def tokenize(text):
    return [word for word in word_tokenize(text.lower()) if not is_number(word)]


# https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
def is_number(word):
    try:
        float(word)
    except ValueError:
        return False
    return True


def split_data(data):
    random.shuffle(data)

    num_examples = len(data)
    print('Number of examples: ', num_examples)
    num_train = int(.6 * num_examples)
    num_val = int(.2 * num_examples)

    return data[:num_train], data[num_train:num_train + num_val], data[num_train + num_val:]


def build_vocabulary(tokenized_data, vocab_size):
    counts = get_token_counts(tokenized_data)
    special_tokens = [PAD_TOKEN, UNK_TOKEN, GO_TOKEN, EOS_TOKEN]
    most_common = [token for token, count in counts.most_common(vocab_size - len(special_tokens))]
    vocab = {token: i for i, token in enumerate(special_tokens + most_common)}
    return vocab


def get_token_counts(tokenized_data):
    counts = Counter()
    for line in chain.from_iterable(tokenized_data):
        counts.update(line)
    return counts


def save_vocab(vocab, file_name):
    inverted_vocab = {vocab_id: token for token, vocab_id in vocab.items()}
    inverted_vocab = [inverted_vocab[vocab_id] for vocab_id in sorted(inverted_vocab)]
    save_list_to_file(inverted_vocab, file_name)


def save_processed_data(data, file_name):
    data_list = [' '.join(line) for line in chain(*data)]
    save_list_to_file(data_list, file_name)


def save_list_to_file(alist, file_name):
    file_path = os.path.join(PROCESSED_PATH, file_name)
    with open(file_path, 'w') as f:
        f.write('\n'.join(alist))


def load_vocab(file_name):
    file_path = os.path.join(PROCESSED_PATH, file_name)
    with open(file_path, 'r') as f:
        vocab = [line.strip() for line in f]
    word2index = {el: i for i, el in enumerate(vocab)}
    index2word = {i: el for i, el in enumerate(vocab)}
    return word2index, index2word


def load_processed_data(file_name):
    file_path = os.path.join(PROCESSED_PATH, file_name)
    with open(file_path, 'r') as f:
        raw_data = [line.strip() for line in f]

    data = []
    for i in range(len(raw_data) - 1):
        qa_pair = (raw_data[i], raw_data[i + 1])
        data.append(qa_pair)
    return data


def get_seqs(file_name):
    vocab, _ = load_vocab(VOCAB_FILE_NAME)
    data = load_processed_data(file_name)

    seqs = []
    for q, a in data:
        q_seq = line_to_seq(q, vocab)
        q_seq.append(vocab[EOS_TOKEN])

        a_seq = [vocab[GO_TOKEN]]
        a_seq.extend(line_to_seq(a, vocab))
        a_seq.append(vocab[EOS_TOKEN])

        qa_pair = (q_seq, a_seq)
        seqs.append(qa_pair)
    return seqs


def line_to_seq(line, vocab):
    return [vocab[token] if token in vocab else vocab[UNK_TOKEN] for token in line.split()]


def indices_to_line(indices, index2word):
    return [index2word[idx] for idx in indices]


def mask_seqs(seqs):
    max_seq_len = max(len(s) for s in seqs)
    vocab, _ = load_vocab(VOCAB_FILE_NAME)
    seq_lens = [len(s) for s in seqs]
    hooded_seqs = [s + [vocab[PAD_TOKEN]] * (max_seq_len - len(s)) for s in seqs]
    masks = [[1] * seq_len + [0] * (max_seq_len - seq_len) for seq_len in seq_lens]
    return hooded_seqs, masks

if __name__ == '__main__':
    print('Getting data...')
    data = get_raw_data()

    print('Tokenizing...')
    data = [(tokenize(q), tokenize(a)) for q, a in data]

    print('Building vocab...')
    vocab = build_vocabulary(data, VOCAB_SIZE)

    print('Splitting data...')
    train, val, test = split_data(data)

    print('Saving data...')
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    save_vocab(vocab, VOCAB_FILE_NAME)

    save_processed_data(train, TRAIN_FILE_NAME)
    save_processed_data(val, VAL_FILE_NAME)
    save_processed_data(test, TEST_FILE_NAME)
