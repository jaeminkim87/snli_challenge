import torch
import torch.nn as nn
import torch.nn.init as init

from model.ops import DiSAN
from model.utils import get_rep_mask


class NN4SNLI(nn.Module):
	def __init__(self, args, data):
		super(NN4SNLI, self).__init__()

		self.d_e = args.d_e
		self.d_h = args.d_h
		self.device = args.device

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim, padding_idx=1)
		self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
		self.word_emb.weight.requires_grad = True

		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		self.dropout = nn.Dropout(args.dropout)
		self.elu = nn.ELU()

		self.disan = DiSAN(args)
		self.fc = nn.Linear()
		self.fc_out = nn.Linear(args.d_h, args.class_size)

		init.xavier_uniform_(self.fc.weight.data)
		init.constant_(self.fc.bias.data, 0)
		init.xavier_uniform_(self.fc_out.weight.data)
		init.constant_(self.fc_out.bias.data, 0)

	def forward(self, batch):
		premise, pre_lengths = batch.premise
		hypothesis, hypo_lengths = batch.hypothesis

		_, p_seq_len = premise.size()
		_, h_seq_len = hypothesis.size()
		p_rep_mask = get_rep_mask(pre_lengths, p_seq_len, self.device)
		h_rep_mask = get_rep_mask(hypt_lengths, h_seq_len, self.device)

		pre_x = self.word_emb(premise)
		hypo_x = self.word_emb(hypothesis)

		pre_s = self.disan(pre_x, p_rep_mask)
		hypo_s = self.disan(hypo_x, h_rep_mask)

		s = torch.cat([pre_s, hypo_s, pre_s - hypo_s, pre_s * hypo_s], dim=-1)

		outs = self.elu(self.fc(self.dropout(s)))
		outs = self.fc_out(self.dropout(outs))
		return outs
