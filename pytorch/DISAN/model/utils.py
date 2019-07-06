import torch
import torch.nn as nn
import torch.nn.init as init

def get_rep_mask(lengths, sentence_len, device):
	req_mask = torch.FloatTensor(len(lengths),sentence_len).to(device)
	req_mask.data.fill_(1)
	for i in range(len(lengths)):
		req_mask[i, lengths[i]:] = 0
	
	return req_mask

def maskted_softmax(vec, mask, dim=1):
	masked_vec = vec * mask.float()
	max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_vec - max_vec)
	masked_sums = masked_exps.sum(dim, keepdim=True)

def get_direct_mask_tile(direction, seq_len, device):
	mask = torch.FloatTensor(seq_len, seq_len).to(device)
	mask.data.fill_(1)
	if direction == 'fw':
		mask = torch.triu(mask, diagonal=1)
	elif direction == 'bw':
		mask = torch.tril(mask, diagonal=-1)
	else:
		raise NotImplementedError('only forward or backward mask is allowed!')
	
	mask.unsqueeze_(0)
	return mask

def get_rep_mask_tile(rep_mask, device):
	batch_size, seq_len = rep_mask.size()
	mask = rep_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

	return mask

