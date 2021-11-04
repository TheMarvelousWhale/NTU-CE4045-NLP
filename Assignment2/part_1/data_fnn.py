import os
import re
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    
    def process_line(self, line):
        return re.sub(r"[^a-zA-Z0-9,\.!?<>\- ]+", '', line).replace('<unk>','').lower()

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                line = self.process_line(line)
                words = line.split() + ['<eos>']
                if words[0]=='<eos>':
                    continue
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids_list = []
            for line in f:
                line = self.process_line(line)
                words = line.split() + ['<eos>']
                ids = []
                if words[0]=='<eos>':
                    continue
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                ids_list.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(ids_list)

        return ids
