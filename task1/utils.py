import torch
import torchtext
from collections import Counter

from tqdm import tqdm
import pandas as pd
import os
import re


def camel_case_split(string):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', string)


def read_pkl(pkl_folder, split, line_limit, word_limit):
    assert split in {'train', 'test'}

    docs = []
    labels = []
    # lens = list()
    word_counter = Counter()
    print(os.path.join(pkl_folder, split + '.pkl'))
    data = pd.read_pickle(os.path.join(pkl_folder, split + '.pkl'))

    for i in tqdm(range(data.shape[0])):
        row = data.iloc[i, :]
        lines, tags = row['lines'], row['labels']

        words = list()
        annotations = list()
        assert len(lines) == len(tags)
        for j, (s, t) in enumerate(zip(lines, tags)):
            if j == line_limit:
                break

            w = s.split()
            # lens.append(len(w))
            w = w[:word_limit]

            if len(w) == 0:
                print("Invalid line:", s)
            else:
                words.append(w)
                word_counter.update(w)
                annotations.append(t)
        # If all lines were empty
        if len(words) == 0:
            continue

        labels.append(annotations)
        docs.append(words)

    return docs, labels, word_counter


def create_input_files(pkl_folder, output_folder, line_limit, word_limit, min_word_count=5, vocab_size=50000):
    # Read training data
    print('\nReading and preprocessing training data...\n')
    os.makedirs(output_folder, exist_ok=True)
    train_docs, train_labels, word_counter = read_pkl(pkl_folder, 'train', line_limit, word_limit)

    # Word2Vec().load('output/word2vec.model')
    vocab = torchtext.vocab.Vocab(word_counter, max_size=vocab_size, min_freq=min_word_count)
    print('\nDiscarding words with counts less than %d, the size of the vocabulary is %d.\n' % (
        min_word_count, len(vocab.stoi)))

    tmaps = {"0": 0, "1": 1, "<pad>": 2, "<start>": 3, "<end>": 4}

    torch.save(vocab, os.path.join(output_folder, 'vocab.pt'))
    print('Vocabulary saved to %s.\n' % os.path.join(output_folder, 'vocab.pt'))

    PAD, UNK = vocab.stoi['<pad>'], vocab.stoi['<unk>']
    # Encode and pad
    print('Encoding and padding training data...\n')
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: vocab.stoi.get(w, UNK), s)) +
            [PAD] * (word_limit - len(s)), doc)) +
                        [[PAD] * word_limit] * (line_limit - len(doc)), train_docs))
    lines_per_train_document = list(map(lambda doc: len(doc), train_docs))
    words_per_train_line = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (line_limit - len(doc)), train_docs))
    train_labels = list(
        map(lambda y: y + [tmaps['<pad>']] * (line_limit - len(y)), train_labels))
    # Save
    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(lines_per_train_document) == len(
        words_per_train_line)
    # Because of the large data, saving as a JSON can be very slow
    torch.save({'docs': encoded_train_docs,
                'labels': train_labels,
                'lines_per_document': lines_per_train_document,
                'words_per_line': words_per_train_line},
               os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Encoded, padded training data saved to %s.\n' % os.path.abspath(output_folder))

    # Free some memory
    del train_docs, encoded_train_docs, train_labels, lines_per_train_document, words_per_train_line

    # Read test data
    print('Reading and preprocessing test data...\n')
    test_docs, test_labels, _ = read_pkl(pkl_folder, 'test', line_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding test data...\n')
    encoded_test_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: vocab.stoi.get(w, UNK), s)) +
            [PAD] * (word_limit - len(s)), doc)) +
                        [[PAD] * word_limit] * (line_limit - len(doc)), test_docs))
    lines_per_test_document = list(map(lambda doc: len(doc), test_docs))
    words_per_test_line = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (line_limit - len(doc)), test_docs))
    test_labels = list(
        map(lambda y: y + [2] * (line_limit - len(y)), test_labels))

    # Save
    print('Saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(lines_per_test_document) == len(
        words_per_test_line)
    torch.save({'docs': encoded_test_docs,
                'labels': test_labels,
                'lines_per_document': lines_per_test_document,
                'words_per_line': words_per_test_line},
               os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Encoded, padded test data saved to %s.\n' % os.path.abspath(output_folder))


def save_checkpoint(epoch, model, optimizer, word_map):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map}
    filename = 'checkpoints/checkpoint_%d.pth.tar' % epoch
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], 1, vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


if __name__ == '__main__':
    create_input_files(pkl_folder='./data',
                       output_folder='./output',
                       line_limit=50,
                       word_limit=20,
                       min_word_count=0)