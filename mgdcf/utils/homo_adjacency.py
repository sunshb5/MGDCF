import numpy as np
import scipy.sparse as sp

def build_homo_adjs(user_item_edges, num_users, num_items, adj_drop_rate):
    # 提取用户-项目边的索引
    user_item_edge_index = user_item_edges.T
    user_item_row, user_item_col = user_item_edge_index
    user_item_adj = sp.csr_matrix((np.ones_like(user_item_row), (user_item_row, user_item_col)),
                                  shape=[num_users, num_items])
    item_user_adj = user_item_adj.T

    # 将邻接矩阵转换为转移矩阵
    def convert_adj_to_trans(adj):
        deg = np.array(adj.sum(axis=-1)).flatten().astype(np.float32)
        inv_deg = np.power(deg, -1)
        inv_deg[np.isnan(inv_deg)] = 0.0
        inv_deg[np.isinf(inv_deg)] = 0.0

        trans = sp.diags(inv_deg) @ adj
        return trans

    # 用户-项目和项目-用户的转移矩阵
    user_item_trans = convert_adj_to_trans(user_item_adj)
    item_user_trans = convert_adj_to_trans(item_user_adj)

    # 将异构图转换为同构图
    def convert_hetero_to_homo(trans_ab, trans_ba):
        homo_trans = trans_ab @ trans_ba
        homo_trans = homo_trans.multiply(homo_trans.T)
        homo_trans.setdiag(0.0)

        homo_trans = homo_trans.tocoo()

        # 筛选并删除低概率边
        probs = homo_trans.data
        probs = probs[probs > 0.0]
        sorted_probs = np.sort(probs)
        threshold = sorted_probs[int(len(probs) * adj_drop_rate)]

        mask = homo_trans.data > threshold
        homo_adj = sp.csr_matrix((np.ones_like(homo_trans.data[mask]), (homo_trans.row[mask], homo_trans.col[mask])),
                                 shape=homo_trans.shape)

        homo_adj = homo_adj.maximum(homo_adj.T)
        homo_adj.eliminate_zeros()

        print(homo_trans.sum(), homo_adj.sum())
        print(len(homo_trans.nonzero()[0]), len(homo_adj.nonzero()[0]))

        return homo_adj

    # 计算用户-用户和项目-项目的同构邻接矩阵
    user_user_adj = convert_hetero_to_homo(user_item_trans, item_user_trans).tocoo()
    item_item_adj = convert_hetero_to_homo(item_user_trans, user_item_trans).tocoo()
    return user_user_adj, item_item_adj

