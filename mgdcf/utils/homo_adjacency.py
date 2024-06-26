import numpy as np
import scipy.sparse as sp


# def normalize_degrees(adj):
#     deg = np.array(adj.sum(axis=-1)).flatten().astype(np.float32)
#     inv_deg = np.power(deg, -1)
#     inv_deg[np.isnan(inv_deg)] = 0.0
#     inv_deg[np.isinf(inv_deg)] = 0.0
#     return sp.diags(inv_deg) @ adj


# def convert_hetero_to_homo(trans_ab, trans_ba, adj_drop_rate):
#     homo_trans = trans_ab @ trans_ba
#     homo_trans = homo_trans.multiply(homo_trans.T)
#     homo_trans.setdiag(0.0)
#     homo_trans = homo_trans.tocoo()

#     probs = homo_trans.data
#     probs = probs[probs > 0.0]
#     sorted_probs = np.sort(probs)
#     threshold = sorted_probs[int(len(probs) * adj_drop_rate)]

#     mask = homo_trans.data > threshold
#     homo_adj = sp.csr_matrix((np.ones_like(homo_trans.data[mask]),
#                               (homo_trans.row[mask], homo_trans.col[mask])),
#                              shape=homo_trans.shape)

#     homo_adj = homo_adj.maximum(homo_adj.T)
#     homo_adj.eliminate_zeros()

#     return homo_adj.tocoo()


# def build_homo_adjs(user_item_edges, num_users, num_items, adj_drop_rate):
#     user_item_adj = sp.csr_matrix((np.ones_like(user_item_edges[0]), user_item_edges),
#                                   shape=(num_users, num_items))
#     item_user_adj = user_item_adj.T

#     user_item_trans = normalize_degrees(user_item_adj)
#     item_user_trans = normalize_degrees(item_user_adj)

#     user_user_adj = convert_hetero_to_homo(user_item_trans, item_user_trans, adj_drop_rate)
#     item_item_adj = convert_hetero_to_homo(item_user_trans, user_item_trans, adj_drop_rate)

#     return user_user_adj, item_item_adj

def build_homo_adjs(user_item_edges, num_users, num_items, adj_drop_rate):
    user_item_edge_index = user_item_edges.T
    user_item_row, user_item_col = user_item_edge_index
    user_item_adj = sp.csr_matrix((np.ones_like(user_item_row), (user_item_row, user_item_col)),
                                  shape=[num_users, num_items])
    item_user_adj = user_item_adj.T

    def convert_adj_to_trans(adj):
        deg = np.array(adj.sum(axis=-1)).flatten().astype(np.float32)
        inv_deg = np.power(deg, -1)
        inv_deg[np.isnan(inv_deg)] = 0.0
        inv_deg[np.isinf(inv_deg)] = 0.0

        trans = sp.diags(inv_deg) @ adj
        return trans

    # def convert_adj_to_trans(adj):
    #     def compute_deg(axis):
    #         deg = np.array(adj.sum(axis=axis)).flatten().astype(np.float32)
    #         inv_sqrt_deg = np.power(deg, -0.5)
    #         inv_sqrt_deg[np.isnan(inv_sqrt_deg)] = 0.0
    #         inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
    #         inv_sqrt_deg = sp.diags(inv_sqrt_deg)
    #         return inv_sqrt_deg
    #
    #     trans = compute_deg(axis=-1) @ adj @ compute_deg(axis=0)
    #     return trans

    user_item_trans = convert_adj_to_trans(user_item_adj)
    item_user_trans = convert_adj_to_trans(item_user_adj)

    def convert_hetero_to_homo(trans_ab, trans_ba):
        homo_trans = trans_ab @ trans_ba
        # homo_trans += homo_trans.T
        homo_trans = homo_trans.multiply(homo_trans.T)
        homo_trans.setdiag(0.0)

        # # renorm
        # homo_trans = convert_adj_to_trans(homo_trans)

        homo_trans = homo_trans.tocoo()

        probs = homo_trans.data
        probs = probs[probs > 0.0]
        sorted_probs = np.sort(probs)
        threshold = sorted_probs[int(len(probs) * adj_drop_rate)]

        mask = homo_trans.data > threshold
        homo_adj = sp.csr_matrix((np.ones_like(homo_trans.data[mask]), (homo_trans.row[mask], homo_trans.col[mask])),
                                 shape=homo_trans.shape)

        homo_adj = homo_adj.maximum(homo_adj.T)
        # homo_adj = homo_adj.minimum(homo_adj.T)
        homo_adj.eliminate_zeros()

        print(homo_trans.sum(), homo_adj.sum())
        print(len(homo_trans.nonzero()[0]), len(homo_adj.nonzero()[0]))
        #
        #
        # deg = homo_adj.sum(axis=-1)
        # degs = np.array(deg, dtype=np.int32).flatten()
        # counter = Counter(degs)
        # # for deg in range(np.max(deg) + 1):
        # #     print(deg, ": ", counter[deg])
        #
        # x = np.arange(0, np.max(deg) + 1)
        # y = [counter[deg] for deg in x]
        #
        # for deg, count in zip(x, y):
        #     print(deg, count)
        # # asdfasdf
        #
        # from matplotlib import pyplot as plt
        # plt.scatter(x, y)
        # # plt.fill_between(x, 0, y)
        # plt.show()
        # asdfasdf

        # homo_adj = sp.csr_matrix((homo_trans.data[mask], (homo_trans.row[mask], homo_trans.col[mask])),
        #                          shape=homo_trans.shape)

        return homo_adj

    user_user_adj = convert_hetero_to_homo(user_item_trans, item_user_trans).tocoo()
    item_item_adj = convert_hetero_to_homo(item_user_trans, user_item_trans).tocoo()
    return user_user_adj, item_item_adj
