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

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(p=args.dropout)
        dim_size = int(args.word_dim) * 2 * 2
        self.dense1 = nn.Linear(self.d_model, args.hidden_size)
        self.dense2 = nn.Linear(self.d_model, 1)
        self.dense3 = nn.Linear(self.d_model, args.classes)
        self.max_pool = nn.MaxPool1d(args.word_max_len)

    def __max_pooling(self, out):
        temp_out = out.permute(0, 2, 1) # [seq, batch, dim] -> [batch, dim, seq]
        output = self.max_pool(temp_out) # [batch, dim, seq = 1]
        u = output.squeeze(2) # [batch, dim, seq = 1] -> [batch, dim]
        return u

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
        #print("prem : {}".format(prem.size()))
        # prem = self.__max_pooling(prem)
        print(prem)

        hypo = self.encoder(target, mask=target_mask, src_key_padding_mask=target_key_padding_mask)
        #print("hypo : {}".format(hypo.size()))
        # hypo = self.__max_pooling(hypo)
        print(hypo)

        prem = prem.view(self.batch_size, self.d_model, -1)
        similarity = prem @ hypo

        #similarity = torch.cat([prem, hypo, torch.abs(prem - hypo), prem * hypo], 1)
        #print("similarity : {}".format(similarity.size()))
        #fc = self.dropout(similarity)
        #print("fc : {}".format(fc.size()))
        #fc = self.dense1(fc)
        #print(fc.size())
        #fc = F.relu(fc, inplace=True)
        #print(fc.size())

        #fc = self.dropout(fc)
        #print(fc.size())
        fc = self.dense2(similarity)
        #print(fc.size())
        fc = fc.view(self.batch_size, -1)
        fc = F.relu(fc, inplace=True)
        #print(fc.size())

        out = self.dense3(fc)
        #print(out.size())
        return out


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
