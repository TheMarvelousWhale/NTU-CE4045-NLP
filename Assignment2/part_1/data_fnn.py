import os
import re
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2count = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word2count[word] = 1
        else:
            self.word2count[word] +=1
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
        
        idss = []
        # Add words to the dictionary and tokenize
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                line = self.process_line(line)
                if line == []:
                    continue
                words = line.split() 
                ids = []
                for word in words:
                    self.dictionary.add_word(word)
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids
