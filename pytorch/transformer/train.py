import pickle
import argparse
import torch
import torch.nn as nn
import nltk
import matplotlib.pyplot as plt

from torch.autograd import Variable
from tqdm import tqdm
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from gluonnlp.data import PadSequence
from model.net import Transformer
from model.data import Tokenizer, Corpus
from model.metric import evaluate, acc
from torch.nn.utils import clip_grad_norm_

from preprocessing.build_data import Preprocessing
from preprocessing.build_vocab import Build_Vocab


# from build_vocab import Build_Vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    # parser.add_argument('--data_type', default='senCNN')
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--classes', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--print_freq', default=3000, type=int)
    # parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--word_dim', default=128, type=int)
    parser.add_argument('--word_max_len', default=30, type=int)
    parser.add_argument('--global_step', default=1000, type=int)
    parser.add_argument('--data_path', default='../data_in')
    parser.add_argument('--file_path', default='../data_in')
    # parser.add_argument('--build_preprocessing', default=False)
    # parser.add_argument('--build_vocab', default=False)

    args = parser.parse_args()
    # p = Preprocessing(args)
    # p.make_datafile()

    v = Build_Vocab(args)
    v.make_vocab()

    with open(args.data_path + '/' + 'vocab.pkl', mode='rb') as io:
        vocab = pickle.load(io)

    padder = PadSequence(length=args.word_max_len, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=nltk.word_tokenize, pad_fn=padder)

    model = Transformer(vocab=vocab, args=args, d_model=args.word_dim)

    tr_ds = Corpus(args.data_path + '/train.txt', tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(args.data_path + '/val.txt', tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    dev_ds = Corpus(args.data_path + '/dev.txt', tokenizer.split_and_transform)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    opt = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_val_loss = 1e+10

    for epoch in tqdm(range(args.epoch), desc='epochs'):
        tr_loss = 0
        tr_acc = 0
        model.train()
        for step, mb in tqdm(enumerate(dev_dl), desc='steps', total=len(dev_dl)):
            sent1, sent2, label = map(lambda elm: elm.to(device), mb)
            opt.zero_grad()
            logits = model(sent1, sent2)
            mb_loss = loss_fn(logits, label)
            mb_loss.backward()
            # clip_grad_norm_(model._fc.weight, 5)
            opt.step()

            with torch.no_grad():
                mb_acc = acc(logits, label)

            tr_loss += mb_loss.item()
            tr_acc += mb_acc.item()

            # if (epoch * len(tr_dl) + step) % args.global_step == 0:
            #     print("epoch : ", epoch + 1, " step : ", step, " loss : ", tr_loss.item(), "acc : ", tr_acc.item())
            #     val_loss = evaluate(model, val_dl, {'loss': loss_fn}, device)['loss']
            #     # writer.add_scalars('loss', {'train': tr_loss / (step + 1),
            #     #                             'val': val_loss}, epoch * len(tr_dl) + step)
            #     model.train()

        else:
            tr_loss /= (step + 1)
            tr_acc /= (step + 1)

            tr_summ = {'loss': tr_loss, 'acc': tr_acc}
            # val_summ = evaluate(model, val_dl, {'loss': loss_fn, 'acc': acc}, device)
            # scheduler.step(val_summ['loss'])
            tqdm.write('epoch : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, tr_summ['loss'],
                                                                        tr_summ['acc']))
            # tqdm.write('epoch : {}, tr_loss: {:.3f}, val_loss: '
            #            '{:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1, tr_summ['loss'],
            #                                                             val_summ['loss'],
            #                                                             tr_summ['acc'], val_summ['acc']))

            # val_loss = val_summ['loss']
            # is_best = val_loss < best_val_loss

            # if is_best:
            #     state = {'epoch': epoch + 1,
            #              'model_state_dict': model.state_dict(),
            #              'opt_state_dict': opt.state_dict()}
            #     summary = {'tr': tr_summ, 'val': val_summ}

            # manager.update_summary(summary)
            # manager.save_summary('summary.json')
            # manager.save_checkpoint(state, 'best.tar')

            #    best_val_loss = val_loss

    # Loss 그래프
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tr_summ['loss'], 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

    # Acc 그래프
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tr_summ['acc'], 'b--')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], val_summ['acc'], 'g--')
    plt.legend(['Tr acc', 'Val acc'])
    plt.show()


if __name__ == '__main__':
    main()
