import torch
from utils import device


def ap_loss_neuralsort_single_user_ori(K, tau, label, pred):
    n = len(label)
    K = min(n, K)
    As1 = torch.sum(torch.abs(pred.reshape(1, -1) - pred.reshape(-1, 1)), dim = 1)
    K_1_to_k = torch.linspace(1, K, K, device=device.get()).reshape(-1, 1)
    P_s_k = torch.softmax(((n + 1 - 2 * K_1_to_k) * pred - As1) / tau, dim=-1)
    label = torch.sum(label * P_s_k[:,:], dim=1) 
    loss = - torch.sum(label * torch.cumsum(label, dim=0) / (1 + torch.arange(K)).to(device.get())) / torch.sum(label)
    return loss





