import torch
from utils import losses, device, update_X_Y, device0
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import sparse


def train_epoch_d(model, optimizer, train_uirdict, K, tau):
	loss = 0
	count = 0
	auc_value, map_value = 0, 0
	for user in train_uirdict.keys():
		count += 1
		items, rels = train_uirdict[user][0], train_uirdict[user][1]
		user, items, rels = torch.tensor(user).to(device.get()), torch.tensor(items).to(device.get()), torch.tensor(rels).to(device.get())
		preds = model(user, items)
		loss_value = losses.ap_loss_neuralsort_single_user_ori(K, tau, rels, preds)
		loss += loss_value
		auc_value += roc_auc_score(rels.detach().cpu().numpy(), preds.detach().cpu().numpy())
		map_value += average_precision_score(rels.detach().cpu().numpy(), preds.detach().cpu().numpy())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return loss / count, torch.true_divide(auc_value, count), torch.true_divide(map_value, count)

