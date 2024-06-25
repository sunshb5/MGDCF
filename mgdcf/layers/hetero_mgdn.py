import tf_sparse as tfs
from grecx.layers import LightGCN
import tensorflow as tf
from mgdcf.utils.normalized_factor import compute_normalized_denominator

class HeteroMGDN(tf.keras.Model):
    def __init__(self, k=10, alpha=0.1, beta=0.9, x_drop_rate=0.0, edge_drop_rate=0.0, 
                 z_drop_rate=0.0, activation=None, kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """
        初始化 HeteroMGDN 模型。

        参数:
        - k (int): 迭代次数。
        - alpha (float): Alpha 参数。
        - beta (float): Beta 参数。
        - x_drop_rate (float): 输入特征的 dropout 率。
        - edge_drop_rate (float): 边的 dropout 率。
        - z_drop_rate (float): 输出特征的 dropout 率。
        - activation (callable): 激活函数。
        - kernel_regularizer: 核正则化器。
        - bias_regularizer: 偏置正则化器。
        """
        super().__init__(*args, **kwargs)
        self.activation = activation  # 激活函数
        self.k = k  # 迭代次数
        self.alpha = alpha  # alpha 参数
        self.beta = beta  # beta 参数
        self.theta = compute_normalized_denominator(alpha, beta, k)  # 计算归一化因子

        self.x_drop_rate = x_drop_rate  # 输入特征 dropout 率
        self.edge_drop_rate = edge_drop_rate  # 边的 dropout 率
        self.z_drop_rate = z_drop_rate  # 输出特征 dropout 率

        self.x_dropout = tf.keras.layers.Dropout(x_drop_rate)  # 输入特征 dropout 层
        self.z_dropout = tf.keras.layers.Dropout(z_drop_rate)  # 输出特征 dropout 层

        self.kernel_regularizer = kernel_regularizer  # 核正则化器
        self.bias_regularizer = bias_regularizer  # 偏置正则化器

    def call(self, inputs, cache=None, training=None, mask=None):
        """
        执行模型的前向传播。

        参数:
        - inputs: 包含输入数据的元组。如果 len(inputs) == 3，则期望 (x, edge_index, edge_weight)。
                  否则，期望 (x, edge_index)。
        - cache: 用于存储中间结果的可选缓存。
        - training: 表示模型是否处于训练模式的布尔标志。
        - mask: 可选的掩码张量，用于屏蔽特定元素。

        返回:
        - output: 经过模型操作后的处理输出张量。
        """
        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        num_nodes = tfs.shape(x)[0]  # 获取节点数
        normed_sparse_adj = LightGCN.norm_adj(edge_index, num_nodes=num_nodes, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)  # 归一化稀疏邻接矩阵并进行 dropout

        h = self.x_dropout(x, training=training)  # 输入特征进行 dropout
        output = h  # 初始输出为输入特征

        for i in range(self.k):
            output = normed_sparse_adj @ output  # 矩阵乘法
            output = output * self.beta + h * self.alpha  # 更新输出特征

        if self.activation is not None:
            output = self.activation(output)  # 应用激活函数

        output /= self.theta  # 归一化输出特征
        output = self.z_dropout(output, training=training)  # 输出特征进行 dropout

        return output  # 返回最终输出特征


