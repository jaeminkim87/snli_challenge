import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules as Modules

from model.utils import _reset_parameters
from model.ops import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 custom_encoder=None, custom_decoder=None, vocab=None, args=None):
        super(Transformer, self).__init__()

        self.batch_size = args.batch_size
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=args.word_dim)

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            encoder_norm = Modules.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            decoder_norm = Modules.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        _reset_parameters(self)
        self.d_model = d_model
        self.len_vocab = len(vocab)
        self.nhead = nhead

        self.dropout1 = nn.Dropout(p=args.dropout)
        self.layout1 = nn.Linear(args.word_max_len * self.d_model * 4, args.hidden_size * 2)
        self.dropout2 = nn.Dropout(p=args.dropout)
        self.layout2 = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.layout3 = nn.Linear(args.hidden_size, args.classes)

    def forward(self, src, target, src_mask=None, target_mask=None,
                memory_mask=None, src_key_padding_mask=None, target_key_padding_mask=None,
                memory_key_padding_mask=None):
        src = self.embedding(src)
        target = self.embedding(target)

        if src.size(1) != target.size(1):
            raise RuntimeError("the batch number of src and target must be equal")

        if src.size(2) != self.d_model or target.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and target must be equal to d_model")

        prem = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask) # [256, 30, 128]
        hypo = self.encoder(target, mask=target_mask, src_key_padding_mask=target_key_padding_mask)

        prem = prem.view(prem.size()[0], -1)
        hypo = hypo.view(hypo.size()[0], -1)

        output = torch.cat([prem, hypo, torch.abs(prem - hypo), prem * hypo], 1)

        output = self.dropout1(output)
        output = F.relu(self.layout1(output))
        output = self.dropout2(output)
        output = F.relu(self.layout2(output))
        output = self.layout3(output)
        return output

        # return out

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
