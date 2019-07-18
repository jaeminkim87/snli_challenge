import pickle
import argparse
import torch
import torch.nn as nn

from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from gluonnlp.data import PadSequence
from model.net import Transformer


# from build_preprocessing import Preprocessing
# from build_vocab import Build_Vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--data_type', default='senCNN')
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--print_freq', default=3000, type=int)
    # parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--word_dim', default=8, type=int)
    parser.add_argument('--word_max_len', default=300, type=int)
    parser.add_argument('--global_step', default=1000, type=int)
    parser.add_argument('--data_path', default='../data_in')
    parser.add_argument('--file_path', default='../nsmc-master')
    # parser.add_argument('--build_preprocessing', default=False)
    # parser.add_argument('--build_vocab', default=False)

    args = parser.parse_args()
    # p = Preprocessing(args)
    # p.makeProcessing()

    # v = Build_Vocab(args)
    # v.make_vocab()

    with open(args.data_path + '/' + 'vocab_char.pkl', mode='rb') as io:
        vocab = pickle.load(io)

    model = Transformer()

if __name__ == '__main__':
    main()
