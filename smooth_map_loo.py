import numpy as np
import torch
from utils import model, device, optimizers, train, evaluation, preprocess, data_utils
import argparse
import copy
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='Parameter settings')
parser.add_argument('--trainpath', nargs='?', default='/home/data/citeulike-a/citeulike_train.csv', help='traindata path.') 
parser.add_argument('--testpath', nargs='?', default='/home/data/citeulike-a/citeulike_test.csv', help='testdata path.') 
parser.add_argument('--meta_info_path', nargs='?', default='/home/data/citeulike-a/citeulike_meta_info.json', help='meta info path.') 
parser.add_argument('--threshold', type=int, default=0, help='binary threshold for pos/neg')
parser.add_argument('--frac', type=float, default=4.0, help='negative sampling ratio')
parser.add_argument('--test_num_neg', type=int, default=99, help='test negative items')
parser.add_argument('--emb_size', type=int, default=8, help='latent factor embedding size (default: 8)')
parser.add_argument('--reg', type=float, default=0.001, help='l2 regularization')
parser.add_argument('--max_iter', type=int, default=1, help='number of epochs to train FW (default: 1)')
parser.add_argument('--zeta_step', type=float, default=0.05, help='decreasing step of zeta (default: 0.1)')
parser.add_argument('--K', type=int, default=10, help='cut off value of NDCG@K (default: 10)')
parser.add_argument('--tau', type=float, default=0.1, help='temperature parameter')
parser.add_argument('--random_range', type=float, default=0.01, help='[-random_range, random_range] for initialization')
parser.add_argument("--cut_offs", type=list, default=[2, 4, 6, 8, 10, 15, 20])


args = parser.parse_args()
trainpath = args.trainpath
testpath = args.testpath
meta_info_path = args.meta_info_path
threshold = args.threshold
frac = args.frac
test_num_neg = args.test_num_neg
emb_size = args.emb_size
reg = args.reg
max_iter = args.max_iter
zeta_step = args.zeta_step
K = args.K
tau = args.tau
cut_offs = args.cut_offs
random_range = args.random_range

print('model parameters: ', args)

print('-------------------Preparing data-------------------')
n_user, n_item, train_udict, test_data, user_set, item_set = data_utils.load_data(trainpath, testpath, meta_info_path)
train_uirdict = preprocess.train_preparation(train_udict, frac, item_set)
test_dataset = data_utils.BPRData(test_data, n_item, train_mat=None, num_ng=0, is_training=False)
test_loader = DataLoader(test_dataset, batch_size=test_num_neg+1, shuffle=False, num_workers=0)

model = model.MF(max(user_set)+1, max(item_set)+1, emb_size=emb_size).to(device.get())

opt = optimizers.GNCCP(model.parameters(), max_iter=max_iter, zeta_step=zeta_step, val_1=-1, val_2=1, converge_eps=0.000001, convex=False)

best_auc = 0.
best_model = None
best_epoch = -1

print('-------------------Training-------------------')
count = 0
while not opt.optimized():
	zeta = opt.zeta
	model = model.to(device.get())
	train_loss, train_auc, train_map = train.train_epoch_d(model, opt, train_uirdict, K, tau)
	best_auc = train_auc
	print('zeta = %.2f, train_loss = %.4f, train_auc = %.5f, train_map = %.5f' %(zeta, train_loss, train_auc, train_map))
	if (count + 1) % max_iter == 0:
		test_ndcg, test_recall, test_map, test_auc, test_mrr = evaluation.calmetrics(model, test_loader, K, cut_offs, test_num_neg)
		print('test ndcg = ', test_ndcg)
		print('test recall = ', test_recall)
		print('test map = ', test_map)
		print('test AUC = ', test_auc)
		print('test MRR = ', test_mrr)
	count += 1
best_model = copy.deepcopy(model)
save_path = '/home/Smooth_AP_Discrete/save_model/citeulike/discrete_' + str(emb_size) + '_' + str(K) + '_' + str(zeta_step) +'_' + str(max_iter) + '_' + str(tau) +'.pkl'
torch.save(best_model, save_path)

print('-------------------Testing-------------------')

test_ndcg, test_recall, test_map, test_auc, test_mrr = evaluation.calmetrics(best_model, test_loader, K, cut_offs, test_num_neg)
print('best epoch = ', best_epoch, 'best auc = ', best_auc)
print('test ndcg = ', test_ndcg)
print('test recall = ', test_recall)
print('test map = ', test_map)
print('test AUC = ', test_auc)
print('test MRR = ', test_mrr)
