import numpy as np
import pandas as pd
from collections import defaultdict
import os
import json
import torch.utils.data as data
import scipy.sparse as sp

def load_all(trainpath, testpath, test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(trainpath, sep=',', header=None, names=['user', 'item'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	test_data = []
	with open(testpath, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split(',')
			# u = eval(arr[0])[0]
			u = int(arr[0])
			pos_i = int(arr[1])
			# print(u, pos_i)
			test_data.append([u, pos_i])
			for i in arr[2:]:
				test_data.append([u, int(i)])
			line = fd.readline()
	return train_data, test_data, user_num, item_num, train_mat

def load_data(trainpath, testpath, meta_info_path):
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
		u = line[1]
		i = line[2]
		user_set.add(u)
		item_set.add(i)
		train_udict[u].append(i)

	test_data = []
	with open(testpath, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split(',')
			u = int(arr[0])
			pos_i = int(arr[1])
			test_data.append([u, pos_i])
			for i in arr[2:]:
				test_data.append([u, int(i)])
			line = fd.readline()

	return n_user, n_item, train_udict, test_data, user_set, item_set

def load_testdata(testpath):
	test_data = []
	with open(testpath, 'r') as fd:
		line = fd.readline()
		while line != None and line != '':
			arr = line.split(',')
			u = int(arr[0])
			pos_i = int(arr[1])
			test_data.append([u, pos_i])
			for i in arr[2:]:
				test_data.append([u, int(i)])
			line = fd.readline()

	return test_data

class BPRData(data.Dataset):
	def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.num_item = num_item
		self.train_mat = train_mat
		self.num_ng = num_ng
		self.is_training = is_training

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_fill = []
		for x in self.features:
			u, i = x[0], x[1]
			for t in range(self.num_ng):
				j = np.random.randint(self.num_item)
				while (u, j) in self.train_mat:
					j = np.random.randint(self.num_item)
				self.features_fill.append([u, i, j])

	def __len__(self):
		return self.num_ng * len(self.features) if self.is_training else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training else self.features

		user = features[idx][0]
		item_i = features[idx][1]
		item_j = features[idx][2] if self.is_training else features[idx][1]
		return user, item_i, item_j