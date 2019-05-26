import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from torch import optim
from mecab import MeCab
from gluonnlp.data import PadSequence
from tqdm import tqdm
from model.data import Corpus
from model.net import Net


def main():
    test_path = Path.cwd() / 'data_in' / 'test.txt'
    vocab_path = Path.cwd() / 'data_in' / 'vocab.pkl'

    with open(vocab_path, mode='rb') as io:
        vocab = pickle.load(io)

    tokenizer = MeCab()
    padder = PadSequence(length=70, pad_val=vocab.token_to_idx['<pad>'])
    test_ds = Corpus(test_path, vocab, tokenizer, padder)
    test_dl = DataLoader(test_ds, batch_size=1024)

    model = Net(vocab_len=len(vocab))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1):
        model.train()
        index = 0
        acc = 0
        for label, sen1, sen2 in tqdm(test_dl, disable=True):
            optimizer.zero_grad()

            pre_label = model(sen1, sen2)

            loss = loss_fn(pre_label, label)
            loss.backward()
            optimizer.step()

            pred_cls = pre_label.data.max(1)[1]
            acc += pred_cls.eq(label.data).cpu().sum()

            print("epoch: {}, index: {}, loss: {}".format((epoch + 1), index, loss.item()))
            index += len(label)

        print('Accuracy : %d %%' % (
                100 * acc / index))


if __name__ == "__main__":
    main()
