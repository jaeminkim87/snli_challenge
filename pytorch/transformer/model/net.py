import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules as Modules

from model.utils import _reset_parameters
from model.ops import TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder


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

        self.dropout = nn.Dropout(p=args.dropout)
        self.dense2 = nn.Linear(self.d_model, 1)
        self.dense3 = nn.Linear(self.d_model, args.classes)
        self.max_pool = nn.MaxPool1d(args.word_max_len)

        self.dense_1 = nn.Linear(128, 1)
        self.dense_3 = nn.Linear(128, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, src, target, src_mask=None, target_mask=None,
                memory_mask=None, src_key_padding_mask=None, target_key_padding_mask=None,
                memory_key_padding_mask=None):
        src = self.embedding(src)
        target = self.embedding(target)

        if src.size(1) != target.size(1):
            raise RuntimeError("the batch number of src and target must be equal")

        if src.size(2) != self.d_model or target.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and target must be equal to d_model")


        prem = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        hypo = self.encoder(target, mask=target_mask, src_key_padding_mask=target_key_padding_mask)

        prem = prem.view(self.batch_size, self.d_model, -1)
        similarity = prem @ hypo

        similarity = self.dense_1(similarity)
        similarity = similarity.view(self.batch_size, -1)
        similarity = self.relu(similarity)

        out = self.dense_3(similarity)

        return out


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
