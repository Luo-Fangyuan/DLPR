import torch
import torch.nn as nn
from utils import LBSign


class MF(nn.Module):
	def __init__(self, n_user, n_item, emb_size, init_range=0.01, weight_user=None, weight_item=None):

		super(MF, self).__init__()
		self.user_emb = nn.Embedding(n_user, emb_size)
		self.item_emb = nn.Embedding(n_item, emb_size)


	def forward(self, userID, itemID):
		user = self.user_emb(userID)
		items = self.item_emb(itemID)
		return torch.sigmoid((user * items).sum(dim=-1))


