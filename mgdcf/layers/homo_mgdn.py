import tf_sparse as tfs
from tf_geometric.nn.conv.gcn import gcn_build_cache_for_graph, gcn_norm_adj
import tensorflow as tf
from mgdcf.utils.normalized_factor import compute_normalized_denominator


class HomoMGDN(tf.keras.Model):

    def __init__(self,
                 k=10,
                 alpha=0.1,
                 beta=0.9,
                 x_drop_rate=0.0,
                 edge_drop_rate=0.0,
                 z_drop_rate=0.0,
                 activation=None,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.activation = activation
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.theta = compute_normalized_denominator(alpha, beta, k)

        self.x_drop_rate = x_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.z_drop_rate = z_drop_rate

        self.x_dropout = tf.keras.layers.Dropout(x_drop_rate)
        self.z_dropout = tf.keras.layers.Dropout(z_drop_rate)

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernels = []
        self.biases = []

    def build_cache_for_graph(self, graph, override=False):
        gcn_build_cache_for_graph(graph, override=override)

    def call(self, inputs, cache=None, training=None, mask=None):
        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        num_nodes = tfs.shape(x)[0]
        sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
        normed_sparse_adj = gcn_norm_adj(sparse_adj, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)

        h = self.x_dropout(x, training=training)
        output = h
        for i in range(self.k):
            output = normed_sparse_adj @ output
            output = output * self.beta + h * self.alpha

        if self.activation is not None:
            output = self.activation(output)

        output /= self.theta
        output = self.z_dropout(output, training=training)

        return output
