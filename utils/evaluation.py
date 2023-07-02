# -*- coding: utf-8 -*-

import numpy as np
import torch
from utils import device, metrics
from sklearn.metrics import roc_auc_score



def calmetrics(model, test_loader, K, cut_offs, test_num_neg):
	ndcg = np.zeros(K)
	recall_matrix = np.zeros(len(cut_offs))
	map_matrix = np.zeros(len(cut_offs))
	auc_value = 0
	mrr_value = 0
	count = 0

	for user, item_i, item_j in test_loader:
		count += 1
		user = user.to(device.get())
		item_i = item_i.to(device.get())

		prediction_i = model(user, item_i)
		prediction_i = prediction_i.detach().cpu().numpy()
		label = [1] + [0] * test_num_neg
		recall, ap, mrr = metrics.map_recall_at_k_multileveltobinary(np.array(label), prediction_i, cut_offs)
		recall_matrix += recall
		map_matrix += ap
		mrr_value += mrr
		ndcg += metrics.normalized_discounted_cumulative_gain_matrix(K, np.array(label), prediction_i)
		auc_value += roc_auc_score(label, prediction_i)
	return ndcg / count, recall_matrix / count, map_matrix / count, auc_value / count, mrr_value / count

