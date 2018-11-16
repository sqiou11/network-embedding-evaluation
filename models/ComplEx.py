import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class ComplEx(nn.Module):
	def __init__(self,embedding_file):
		super(ComplEx,self).__init__()
		self.ent_re_embeddings=nn.Embedding(14541,128)
		self.ent_im_embeddings=nn.Embedding(14541,128)
		self.rel_re_embeddings=nn.Embedding(237,128)
		self.rel_im_embeddings=nn.Embedding(237,128)
		self.load_state_dict(torch.load(embedding_file))
		self.eval()

	#score function of ComplEx
	def _calc(self,e_re_h,e_im_h,e_re_t,e_im_t,r_re,r_im):
		return torch.sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t,1,False)

	def predict(self, predict_h, predict_t, predict_r):
		predict_r = torch.LongTensor([predict_r] * predict_h.size()[0])
		p_re_h=self.ent_re_embeddings(Variable(predict_h))
		p_re_t=self.ent_re_embeddings(Variable(predict_t))
		p_re_r=self.rel_re_embeddings(Variable(predict_r))
		p_im_h=self.ent_im_embeddings(Variable(predict_h))
		p_im_t=self.ent_im_embeddings(Variable(predict_t))
		p_im_r=self.rel_im_embeddings(Variable(predict_r))
		p_score = self._calc(p_re_h, p_im_h, p_re_t, p_im_t, p_re_r, p_im_r)
		return p_score.cpu().data.numpy().tolist()
