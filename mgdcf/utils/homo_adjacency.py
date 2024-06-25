import numpy as np
import scipy.sparse as sp


def normalize_degrees(adj):
    deg = np.array(adj.sum(axis=-1)).flatten().astype(np.float32)
    inv_deg = np.power(deg, -1)
    inv_deg[np.isnan(inv_deg)] = 0.0
    inv_deg[np.isinf(inv_deg)] = 0.0
    return sp.diags(inv_deg) @ adj


def convert_hetero_to_homo(trans_ab, trans_ba, adj_drop_rate):
    homo_trans = trans_ab @ trans_ba
    homo_trans = homo_trans.multiply(homo_trans.T)
    homo_trans.setdiag(0.0)
    homo_trans = homo_trans.tocoo()

    probs = homo_trans.data
    probs = probs[probs > 0.0]
    sorted_probs = np.sort(probs)
    threshold = sorted_probs[int(len(probs) * adj_drop_rate)]

    mask = homo_trans.data > threshold
    homo_adj = sp.csr_matrix((np.ones_like(homo_trans.data[mask]),
                              (homo_trans.row[mask], homo_trans.col[mask])),
                             shape=homo_trans.shape)

    homo_adj = homo_adj.maximum(homo_adj.T)
    homo_adj.eliminate_zeros()

    return homo_adj.tocoo()


def build_homo_adjs(user_item_edges, num_users, num_items, adj_drop_rate):
    user_item_adj = sp.csr_matrix((np.ones_like(user_item_edges[0]), user_item_edges),
                                  shape=(num_users, num_items))
    item_user_adj = user_item_adj.T

    user_item_trans = normalize_degrees(user_item_adj)
    item_user_trans = normalize_degrees(item_user_adj)

    user_user_adj = convert_hetero_to_homo(user_item_trans, item_user_trans, adj_drop_rate)
    item_item_adj = convert_hetero_to_homo(item_user_trans, user_item_trans, adj_drop_rate)

    return user_user_adj, item_item_adj
