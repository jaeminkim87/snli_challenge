import pickle
import argparse
import torch
import torch.nn as nn
import nltk
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from gluonnlp.data import PadSequence
from model.net import Transformer, ScheduledOptim
from model.data import Tokenizer, Corpus
from model.metric import evaluate, acc
from torch.nn.utils import clip_grad_norm_

from preprocessing.build_data import Preprocessing
from preprocessing.build_vocab import Build_Vocab


# from build_vocab import Build_Vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    # parser.add_argument('--data_type', default='senCNN')
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--classes', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--print_freq', default=3000, type=int)
    # parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--word_dim', default=64, type=int)
    parser.add_argument('--word_max_len', default=30, type=int)
    parser.add_argument('--global_step', default=1000, type=int)
    parser.add_argument('--data_path', default='../data_in')
    parser.add_argument('--file_path', default='../data_in')
    # parser.add_argument('--build_preprocessing', default=False)
    # parser.add_argument('--build_vocab', default=False)

    args = parser.parse_args()
    # p = Preprocessing(args)
    # p.make_datafile()

    # v = Build_Vocab(args)
    # v.make_vocab()

    with open(args.data_path + '/' + 'vocab.pkl', mode='rb') as io:
        vocab = pickle.load(io)

    padder = PadSequence(length=args.word_max_len, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=nltk.word_tokenize, pad_fn=padder)

    model = Transformer(vocab=vocab, args=args, d_model=args.word_dim)

    tr_ds = Corpus(args.data_path + '/train.txt', tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(args.data_path + '/val.txt', tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    test_ds = Corpus(args.data_path + '/test.txt', tokenizer.split_and_transform)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    dev_ds = Corpus(args.data_path + '/dev.txt', tokenizer.split_and_transform)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # opt = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    opt = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        args.word_dim, 4000)
    loss_fn = nn.CrossEntropyLoss()
    # scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_val_loss = 1e+10

    epochs = []
    valid_score = []
    test_score = []

    min_loss = 100
    max_dev = 0
    max_test = 0

    for epoch in tqdm(range(args.epoch), desc='epochs'):
        tr_loss = 0
        tr_acc = 0
        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='tr', total=len(tr_dl)):
            sent1, sent2, label = map(lambda elm: elm.to(device), mb)
            opt.zero_grad()
            logits = model(sent1, sent2)
            _, indices = logits.max(1)
            mb_loss = loss_fn(logits, label)
            mb_loss.backward()
            # clip_grad_norm_(model.layout3.weight, 5)
            # opt.step()
            opt.step_and_update_lr()
            tr_loss += mb_loss.item()
        train_loss = tr_loss / len(tr_dl)
        model.eval()

        valid_match = []
        for step, mb in tqdm(enumerate(val_dl), desc='val', total=len(val_dl)):
            sent1, sent2, label = map(lambda elm: elm.to(device), mb)
            logits = model(sent1, sent2)
            _, indices = logits.max(1)
            match = torch.eq(indices, label).detach()
            valid_match.extend(match.cpu())
        valid_accuracy = np.sum(valid_match) * 100 / len(valid_match)
        valid_score.append(valid_accuracy)

        min_loss = min(min_loss, train_loss)
        max_dev = max(max_dev, valid_accuracy)

        tqdm.write("[%2d], loss: %.3f, val_acc: %.3f" % (epoch + 1, train_loss, valid_accuracy))

        #
        # test evaluate
        #
        test_match = []
        for step, mb in tqdm(enumerate(test_dl), desc='val', total=len(test_dl)):
            sent1, sent2, label = map(lambda elm: elm.to(device), mb)
            logits = model(sent1, sent2)
            _, indices = logits.max(1)
            match = torch.eq(indices, label).detach()
            test_match.extend(match.cpu())
        test_accuracy = np.sum(test_match) * 100 / len(test_match)
        test_score.append(test_accuracy)


        max_test = max(max_test, test_accuracy)

        tqdm.write("[%2d], test_acc: %.3f" % (epoch + 1, test_accuracy))


if __name__ == '__main__':
    main()
