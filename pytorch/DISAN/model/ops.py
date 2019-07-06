import torch
import torch.nn as nn
import torch.nn.init as init

from model.utils import *

class Source2Token(nn.Module):
	def __init__(self, d_h, dropout=0.2):
		super(Source2Token, self).__init__()

		self.d_h = d_h
		self.dropout_rate = dropout

		self.fc1 = nn.Linear(d_h, d_h)
		self.fc2 = nn.Linear(d_h, d_h)

		init.xavier_uniform_(self.fc1.weight.data)
		init.constant_(self.fc1.bias.data, 0)
		init.xavier_uniform_(self.fc2.weight.data)
		init.constant_(self.fc2.bias.data, 0)

		self.elu = nn.ELU()
		self.softmax = nn.Softmax(dim=-2)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, rep_mask):
		x = self.dropout(x)
		map1 = self.elu(self.fc1(x))
		map2 = self.fc2(self.dropout(map1))

		soft = masked_softmax(map2, rep_mask.unsqueeze(-1), dim=1)
		out = torch.sum(x * soft, dim=1)

		return out

class DiSA(nn.Module):
	def __init__(self, args, direction):
		super(DiSA, self).__init__()

		self.d_e = args.d_e
		self.d_h = args.d_h
		self.direction = direction
		self.dropout_rate = args.dropout
		self.device = args.deivce

		self.fc = nn.Linear(args.d_e, args.d_h)
		init.xavier_unifor_(self.fc.weight.data)
		init.constant_(self.fc.bias.data, 0)

		self.w_1 = nn.Linear(args.d_h, args.d_h)
		self.w_2 = nn.Linear(args.d_h, args.d_h)
		init.xavier_uniform_(self.w_1.weight)
		init.xavier_uniform_(self.w_2.weight)
		init.constant_(self.w_1.bias, 0)
		init.constant_(self.w_2.bias, 0)
		self.w_1.bias.requires_grad = False
		self.w_2.bias.requires_grad = False

		self.b_1 = nn.Parameter(torch.zeros(args.d_h))
		self.c = nn.Paramater(torch.Tensor([5.0]), requires_grad=False)

		slef.w_f1 = nn.Linear(args.d_h, args.d_h)
		self.w_f2 = nn.Linear(args.d_h, args.d_h)
		init.xavier_uniform_(self.w_f1.weight)
		init.xavier_uniform_(self.w_f2.weight)
		init.constant_(self.w_f1.bias, 0)
		init.constant_(self.w_f2.bias, 0)
		self.w_f1.bias.requires_grad = False
		self.w_f2.bias.requires_grad = False
		self.b_f = nn.Parameter(torch.zeros(args.d_h))

		self.elu = nn.ELU()
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=-2)
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(args.dropout)

	def forward(self, x, rep_mask):
		batch_size, seq_len, d_e = x.size()
		rep_mask_tile = get_rep_mask_tile(rep_mask, self.device)
		direct_mask_tile = get_direct_mask_tile(self.direction, seq_len, self.device)
		mask = rep_mask_tile * direct_mask_tile
		mask.unsqueeze_(-1)

		x_dp = self.deropout(x)
		rep_mal = self.elu(self.fc(x_dp))
		rep_map_tile = rep_map.unsqueeze(1).expand(batch_size, seq_len, seq_len, d_e)
		req_map_dp = self.dropout(rep_map)

		dependent_etd = self.w_1(erp_map_dp).unsqueeze(1)
		head_etd = self.w_2(rep_map_dp).unsqueeze(2)

		logits = self.c * self.tanh((dependent_etd + head_etd + self.b_1) / self.c)

		attn_score = masked_softmax(logits, mask, dim=2)
		attn_score = attn_score * mask

		attn_result = torch.sum(attn_score * rep_map_tile, dim=2)

		fusion_gate = self.sigmoid(self.w_f1(self.dropout(rep_map)) + self.w_f2(self.dropout(attn_result)) + self.b_f)
		out = fusion_gate * rep_map + (1 - fusion_gate) * attn_result

		out = out * rep_mask.unsqueeze(-1)

		return out

class DiSAN(nn.Module):
	def __init__(self, args):
		super(DiSAN, self).__init__()
		self.d_e = args.d_e
		self.d_h = args.d_h
		self.debice = args.device

		self.fw_DiSA = DiSA(args, direction='fw')
		self.bs_DiSA = DiSA(args, direction='bw')

		self. source2token = Source2Token(args.d_h * 2, args.dropout)

	def forward(self, inputs, rep_mask):
		for_u = self.fw_DiSA(inputs, rep_mask)
		back_u = self.bw_DiSA(inputs, rep_mask)

		u = torch.cat([for_u, back_u], dim=-1)

		s = self.source2token(u, rep_mask)

		return s





