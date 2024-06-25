import os
import pickle
from grecx.layers import LightGCN


def build_cache_for_graph(graph, override=False):
    if override:
        graph.cache[LightGCN.CACHE_KEY] = None
    LightGCN.norm_adj(graph.edge_index, graph.num_nodes, cache=graph.cache)


def load_cache(path, func):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        data = func()
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        return data
