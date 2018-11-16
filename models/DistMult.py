import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os

class DistMult(nn.Module):
	def __init__(self, embedding_file):
		embedding_file = os.path.abspath(embedding_file)
		super(DistMult, self).__init__()
		self.ent_embeddings=nn.Embedding(14541, 128)
		self.rel_embeddings=nn.Embedding(237, 128)
		self.load_state_dict(torch.load(embedding_file))
		self.eval()
	
	# score function of DistMult
	def _calc(self, h, t, r):
		return torch.sum(h*t*r,1,False)

	# either predict_h is an array of the same node and predict_t is an array of
	# all nodes, or vice versa, for a given test case
	def predict(self, predict_h, predict_t, r):
		predict_r = torch.LongTensor([r] * predict_h.size()[0])
		#p_e_h = self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cpu())
		#p_e_t = self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cpu())
		#p_e_r = self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cpu())
		p_e_h = self.ent_embeddings(Variable(predict_h))
		p_e_t = self.ent_embeddings(Variable(predict_t))
		p_e_r = self.rel_embeddings(Variable(predict_r))
		p_score = self._calc(p_e_h,p_e_t,p_e_r)
		return p_score.cpu().data.numpy().tolist()
