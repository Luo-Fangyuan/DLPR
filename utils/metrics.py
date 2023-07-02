import numpy as np
import math


def ideal_discounted_cumulative_gain_matrix(K: int, label: np.ndarray):
    lrank = label.argsort()[::-1]
    idcg_matrix = np.zeros(K)
    idcg_list = [(2**label[lrank[r]] - 1) / math.log2(r + 2) for r in range(min(K, len(label)))]
    for r in range(1, K+1):
        idcg_matrix[r-1] = sum(idcg_list[:min(r, len(label))])
    return idcg_matrix

def normalized_discounted_cumulative_gain_matrix(K: int, label: np.ndarray, pred: np.ndarray):
    assert len(label) == len(pred)
    dcg_matrix = np.zeros(K)
    prank = pred.argsort()[::-1]
    dcg_list = [(2**label[prank[r]] - 1) / math.log2(r + 2) for r in range(min(K, len(label)))]
    for r in range(1, K+1):
        dcg_matrix[r-1] = sum(dcg_list[:min(r, len(label))])
    return dcg_matrix / ideal_discounted_cumulative_gain_matrix(K, label)

def map_recall_at_k_multileveltobinary(groundtruth, pred, cut_offs):
    prank = pred.argsort()[::-1]
    recall_value = np.zeros(len(cut_offs))
    map_value = np.zeros(len(cut_offs))
    rank_gt = np.array([groundtruth[prank[i]] for i in range(len(prank))])
    mrr_value = max(rank_gt / np.arange(1, len(rank_gt) + 1))
    pos_items = np.sum(rank_gt)
    for j_cutoff in range(len(cut_offs)):
        cut_off = min(cut_offs[j_cutoff], len(groundtruth))
        recall_value[j_cutoff] = np.sum(rank_gt[:cut_off]) / pos_items
        p_value = (rank_gt[:cut_off] * np.cumsum(rank_gt[:cut_off])) / (1 + np.arange(cut_off))
        p_value = p_value[np.nonzero(p_value)].mean() if p_value[np.nonzero(p_value)].size > 0 else 0
        map_value[j_cutoff] = p_value
    return recall_value, map_value, mrr_value


    