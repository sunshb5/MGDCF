import os
import pickle
from grecx.layers import LightGCN

def build_cache_for_graph(graph, override=False):
    """
    构建图的缓存信息。

    参数:
    - graph: 图数据结构。
    - override (bool): 是否覆盖现有缓存信息。
    """
    if override:
        graph.cache[LightGCN.CACHE_KEY] = None  # 清空缓存
    LightGCN.norm_adj(graph.edge_index, graph.num_nodes, cache=graph.cache)  # 计算归一化稀疏邻接矩阵

def load_cache(path, func):
    """
    加载缓存数据或者生成新的缓存数据。

    参数:
    - path (str): 缓存文件路径。
    - func (callable): 生成缓存数据的函数。

    返回:
    - data: 加载或生成的缓存数据。
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # 创建目录
        
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)  # 加载已存在的缓存数据
    else:
        data = func()  # 生成新的缓存数据
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=4)  # 将生成的缓存数据写入文件
        return data  # 返回生成的缓存数据
