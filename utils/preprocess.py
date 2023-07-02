import numpy as np
import pandas as pd
from collections import defaultdict
import os
import json

def load_data(trainpath, testpath, meta_info_path, threshold):
	stat_path = os.path.join(meta_info_path)
	df = pd.read_csv(trainpath, sep=',', header=None) 
	df_test = pd.read_csv(testpath, sep=',', header=None)
	print('train data shape = ', df.shape)
	print('test data shape = ', df_test.shape)


	with open(os.path.join(stat_path), 'r') as f:
		dataset_meta_info = json.load(f)

	n_user = dataset_meta_info['user_size']
	n_item = dataset_meta_info['item_size']

	train_udict = defaultdict(list)
	user_set = set()
	item_set = set()

	for line in df.itertuples():  
		if line[3] >= threshold:
			u = line[1]
			i = line[2]
			user_set.add(u)
			item_set.add(i)
			train_udict[u].append(i)

	test_udict = defaultdict(list)

	for line in df_test.itertuples():
		if line[3] >= threshold:
			u = line[1]
			i = line[2]
			user_set.add(u)
			item_set.add(i)
			test_udict[u].append(i)

	return n_user, n_item, train_udict, test_udict, user_set, item_set

def train_preparation(train_udict, frac, item_set):

	uir_dict = defaultdict(list)

	for user in train_udict.keys():
		pos_items = train_udict[user]
		neg_pool = list(item_set - set(pos_items))
		len_pos = len(pos_items)
		num_neg = int(frac * len_pos)

		train_rel = [1] * len_pos  + [0] * num_neg
		neg_i = list(np.random.choice(neg_pool, size=num_neg, replace=False))
		items = pos_items + neg_i

		uir_dict[user].append(items)
		uir_dict[user].append(train_rel)

	return uir_dict


def test_preparation(train_udict, test_udict, frac, item_set):

	uir_dict = defaultdict(list)

	for user in test_udict.keys():
		pos_items = test_udict[user]
		neg_pool = list(item_set - set(pos_items) - set(train_udict[user])) 
		len_pos = len(pos_items)
		num_neg = int(frac * len_pos)

		test_rel = [1] * len_pos + [0] * num_neg
		neg_i = list(np.random.choice(neg_pool, size=num_neg, replace=False))
		items = pos_items + neg_i

		uir_dict[user].append(items)
		uir_dict[user].append(test_rel)

	return uir_dict