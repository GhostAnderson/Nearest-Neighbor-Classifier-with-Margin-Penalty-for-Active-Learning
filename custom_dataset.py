import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import torchtext
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer

class IMDb(Dataset):
    def __init__(self, max_length=100, mode='train') -> None:
        super().__init__()
        self.mode = mode
        self.dataset = torchtext.datasets.IMDB(split=mode)
        
        self.max_length = max_length

        self._init_dataset()
    
    def _init_dataset(self):
        
        self.raw_text = []
        self.Ys = []            
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        for label, line in self.dataset:
            self.raw_text.append(line)
            self.Ys.append(0 if label == 'neg' else 1)


    def __getitem__(self, index):

        X, Y = self.raw_text[index], self.Ys[index]
        X = self.tokenizer.encode(
            X,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        ).squeeze()

        return X, Y, index

    def __len__(self):
        return len(self.raw_text)

class AG_NEWS(Dataset):
    def __init__(self, max_length=100, mode='train') -> None:
        super().__init__()
        self.mode = mode
        self.dataset = torchtext.datasets.AG_NEWS(split=mode)
        self.max_length = max_length

        self._init_dataset()
    
    def _init_dataset(self):

        self.raw_text = []
        self.Ys = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        for label, line in self.dataset:
            self.raw_text.append(line)
            self.Ys.append(int(label)-1)
        
    
    def __getitem__(self, index):
        X, Y = self.raw_text[index], self.Ys[index]
        X = self.tokenizer.encode(
            X,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        ).squeeze()

        return X, Y, index

    def __len__(self):
        return len(self.raw_text)

class Yelp_Polarity(Dataset):
    def __init__(self, max_length=100, mode='train') -> None:
        super().__init__()
        self.mode = mode
        self.dataset = torchtext.datasets.YelpReviewPolarity(split=mode)
        self.max_length = max_length

        self._init_dataset()
    
    def _init_dataset(self):

        self.raw_text = []
        self.Ys = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        for label, line in self.dataset:
            self.raw_text.append(line)
            self.Ys.append(int(label)-1)
        
    
    def __getitem__(self, index):
        X, Y = self.raw_text[index], self.Ys[index]
        X = self.tokenizer.encode(
            X,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_tensors='pt'
        ).squeeze()

        return X, Y, index

    def __len__(self):
        return len(self.raw_text)


class Telecom(Dataset):
    def __init__(self, mode='train', max_length = 100) -> None:
        super().__init__()
        if mode == 'train':
            filename = 'train.csv'
        elif mode == 'test':
            filename = 'test.csv'
        
        self.max_length = max_length
        self.df = pd.read_csv(filename, header=None)
        self.class_num = 0
        self._init_dataset()
    
    def _init_dataset(self):
        self.raw_text = []
        self.Ys = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

        for index, row in self.df.iterrows():
            self.raw_text.append(row[1])
            self.Ys.append(row[2])
            if row[2] > self.class_num:
                self.class_num = row[2]
        self.class_num += 1
    
    def __getitem__(self, index):
        X, Y = self.raw_text[index], self.Ys[index]
        X = self.tokenizer.encode(
            X,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).squeeze()

        return X, Y, index

    def __len__(self):
        return len(self.raw_text)