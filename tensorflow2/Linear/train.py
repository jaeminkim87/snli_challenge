import tensorflow as tf
import pickle

from pathlib import Path
from mecab import MeCab
from gluonnlp.data import PadSequence
from tqdm import tqdm
from absl import app


def main():
    train_path = Path.cwd() / 'data_in' / 'train.txt'
    val_path = Path.cwd() / 'data_in' / 'val.txt'
    vocab_path = Path.cwd() / 'data_in' / 'vocab.pkl'

    with open(vocab_path, mode='rb') as io:
        vocab = pickle.load(io)
        
    length = 70
    batch_size = 1024
    learning_rate = 0.01
    epochs = 10
        
    tokenizer = MeCab()
    padder = PadSequence(length=length, pad_val=vocab.token_to_idx['<pad>'])
    
    
    
    

if __name__ == '__main__':
    app.run(main)