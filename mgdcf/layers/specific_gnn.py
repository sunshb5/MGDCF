import tensorflow as tf
from grecx.layers import LightGCN
import tf_sparse as tfs


class HeteroGNN(tf.keras.Model):
    """
    Each NGCF Convolutional Layer
    """

    def __init__(self, dense_activation=tf.nn.leaky_relu, edge_drop_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense_activation = dense_activation
        self.gcn_dense = None
        self.interaction_dense = None
        self.edge_drop_rate = edge_drop_rate

    def build(self, input_shape):
        x_shape, _ = input_shape
        self.gcn_dense = tf.keras.layers.Dense(x_shape[1], activation=self.dense_activation)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs[0], inputs[1]

        num_nodes = tfs.shape(x)[0]
        normed_adj = LightGCN.norm_adj(edge_index, num_nodes=num_nodes, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)

        h = normed_adj @ x
        h = self.gcn_dense(h)

        return h


class HeteroGNNSpecific(tf.keras.Model):

    def __init__(self, k=4, z_drop_rate=0.0, edge_drop_rate=0.0, dense_activation=tf.nn.leaky_relu, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hetero_gcns = [HeteroGNN(dense_activation=dense_activation, edge_drop_rate=edge_drop_rate) for _ in range(k)]
        self.dropout = tf.keras.layers.Dropout(z_drop_rate)

        for i, hetero_gcn in enumerate(self.hetero_gcns):
            setattr(self, "hetero_gcn{}".format(i), hetero_gcn)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs

        h = x
        h_list = [h]

        for hetero_gcn in self.hetero_gcns:
            h = hetero_gcn([h, edge_index], training=training, cache=cache)
            h = self.dropout(h, training=training)
            h_list.append(h)

        h = tf.reduce_mean(tf.stack(h_list, axis=0), axis=0)

        return h
