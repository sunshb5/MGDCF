import os
import sys
import time
import logging
import json
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from grecx.layers import LightGCN
from grecx.evaluation.ranking import evaluate_mean_global_metrics
from grecx.datasets.light_gcn_dataset import LightGCNDataset
import tf_geometric as tfg
from tf_geometric.utils import tf_utils
from mgdcf.layers.specific_gnn import HeteroGNNSpecific
from mgdcf.layers.hetero_mgdn import HeteroMGDN
from mgdcf.layers.homo_mgdn import HomoMGDN
from mgdcf.cache import load_cache, build_cache_for_graph
from mgdcf.utils.homo_adjacency import build_homo_adjs

np.set_printoptions(precision=4)
logging.basicConfig(format='%(asctime)s %(message)s\t', level=logging.INFO, stream=sys.stdout)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--emb_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr_decay", type=float, required=True)
    parser.add_argument("--z_l2_coef", type=float, required=True)
    parser.add_argument("--num_negs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--adj_drop_rate", type=float, required=False)
    parser.add_argument("--alpha", type=float, required=False)
    parser.add_argument("--beta", type=float, required=False)
    parser.add_argument("--num_iter", type=int, required=False)
    parser.add_argument("--x_drop_rate", type=float, required=False)
    parser.add_argument("--z_drop_rate", type=float, required=False)
    parser.add_argument("--edge_drop_rate", type=float, required=False)
    args = parser.parse_args()
    logging.info(args)
    return args


def setup_environment(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    return LightGCNDataset(args.dataset).load_data()


class CollaborativeFilteringTask:
    def __init__(self, data_dict, args):
        self.num_users = data_dict["num_users"]
        self.num_items = data_dict["num_items"]
        self.user_item_edges = data_dict["user_item_edges"]
        self.train_index = data_dict["train_index"]
        self.train_user_item_dict = data_dict["train_user_items_dict"]
        self.test_user_item_dict = data_dict["test_user_items_dict"]
        self.train_user_item_edges = self.user_item_edges[self.train_index]
        self.train_user_item_edge_index = self.train_user_item_edges.transpose()
        self.args = args

    def get_model_and_forward(self):
        if self.args.method in ["MF"]:
            return self.build_mf_model()
        elif self.args.method.startswith("Hetero") or self.args.method in ["LightGCN", "APPNP", "JKNet", "DropEdge"]:
            return self.build_hetero_model()
        elif self.args.method.startswith("Homo"):
            return self.build_homo_model()
        else:
            raise ValueError(f"Invalid method name: {self.args.method}")

    def build_mf_model(self):
        user_embeddings = tf.Variable(
            tf.random.truncated_normal([self.num_users, self.args.emb_size], stddev=1.0 / np.sqrt(self.args.emb_size)))
        item_embeddings = tf.Variable(
            tf.random.truncated_normal([self.num_items, self.args.emb_size], stddev=1.0 / np.sqrt(self.args.emb_size)))
        z_dropout = tf.keras.layers.Dropout(self.args.z_drop_rate)

        @tf.function
        def forward(training=False):
            user_hidden = z_dropout(user_embeddings, training=training)
            item_hidden = z_dropout(item_embeddings, training=training)
            return user_hidden, item_hidden

        return forward

    def build_hetero_model(self):
        virtual_graph = tfg.Graph(
            x=tf.Variable(
                tf.random.truncated_normal([self.num_users + self.num_items, self.args.emb_size],
                                           stddev=1 / np.sqrt(self.args.emb_size)),
                name="virtual_embeddings"
            ),
            edge_index=LightGCN.build_virtual_edge_index(self.train_user_item_edge_index, self.num_users)
        )

        if self.args.method in ["HeteroMGDCF", "LightGCN", "APPNP"]:
            model = HeteroMGDN(k=self.args.num_iter, alpha=self.args.alpha, beta=self.args.beta,
                               edge_drop_rate=self.args.edge_drop_rate)
            build_cache_for_graph(virtual_graph)
            x_dropout = tf.keras.layers.Dropout(self.args.x_drop_rate)
            z_dropout = tf.keras.layers.Dropout(self.args.z_drop_rate)

            @tf_utils.function
            def forward(training=False):
                virtual_hidden = x_dropout(virtual_graph.x, training=training)
                virtual_hidden = model([virtual_hidden, virtual_graph.edge_index], training=training, cache=virtual_graph.cache)
                virtual_hidden = z_dropout(virtual_hidden, training=training)
                user_hidden = virtual_hidden[:self.num_users]
                item_hidden = virtual_hidden[self.num_users:]
                return user_hidden, item_hidden

        elif self.args.method in ["JKNet", "DropEdge"]:
            if self.args.method == "JKNet" and self.args.edge_drop_rate > 0.0:
                raise ValueError("JKNet edge_drop_rate must be 0.0")

            model = HeteroGNNSpecific(k=self.args.num_iter, z_drop_rate=self.args.z_drop_rate,
                                      edge_drop_rate=self.args.edge_drop_rate)
            build_cache_for_graph(virtual_graph)
            x_dropout = tf.keras.layers.Dropout(self.args.x_drop_rate)

            @tf_utils.function
            def forward(training=False):
                virtual_hidden = x_dropout(virtual_graph.x, training=training)
                virtual_hidden = model([virtual_hidden, virtual_graph.edge_index], training=training, cache=virtual_graph.cache)
                user_hidden = virtual_hidden[:self.num_users]
                item_hidden = virtual_hidden[self.num_users:]
                return user_hidden, item_hidden

        return forward

    def build_homo_model(self):
        user_embeddings = tf.Variable(
            tf.random.truncated_normal([self.num_users, self.args.emb_size], stddev=1.0 / np.sqrt(self.args.emb_size)))
        item_embeddings = tf.Variable(
            tf.random.truncated_normal([self.num_items, self.args.emb_size], stddev=1.0 / np.sqrt(self.args.emb_size)))

        train_user_item_edges = np.array(self.train_user_item_edges)
        adj_cache_path = os.path.join("cache", f"{self.args.dataset}_adj_{self.args.adj_drop_rate}.p")

        def build_adjs_func():
            return build_homo_adjs(train_user_item_edges, self.num_users, self.num_items,
                                   adj_drop_rate=self.args.adj_drop_rate)
        user_user_adj, item_item_adj = load_cache(adj_cache_path, func=build_adjs_func)

        user_user_edge_index = np.stack([user_user_adj.row, user_user_adj.col], axis=0)
        user_user_edge_weight = user_user_adj.data
        item_item_edge_index = np.stack([item_item_adj.row, item_item_adj.col], axis=0)
        item_item_edge_weight = item_item_adj.data

        user_graph = tfg.Graph(user_embeddings, user_user_edge_index,
                               edge_weight=user_user_edge_weight).convert_edge_to_directed(merge_mode="max")
        item_graph = tfg.Graph(item_embeddings, item_item_edge_index,
                               edge_weight=item_item_edge_weight).convert_edge_to_directed(merge_mode="max")

        if self.args.method == "HomoMGDCF":
            user_model = HomoMGDN(k=0, alpha=self.args.alpha, beta=self.args.beta, edge_drop_rate=self.args.edge_drop_rate)
            item_model = HomoMGDN(k=self.args.num_iter, alpha=self.args.alpha, beta=self.args.beta,
                                  edge_drop_rate=self.args.edge_drop_rate)
            x_dropout = tf.keras.layers.Dropout(self.args.x_drop_rate)
            z_dropout = tf.keras.layers.Dropout(self.args.z_drop_rate)

            user_model.build_cache_for_graph(user_graph)
            item_model.build_cache_for_graph(item_graph)

            @tf.function
            def forward(training=False):
                user_hidden = x_dropout(user_graph.x, training=training)
                item_hidden = x_dropout(item_graph.x, training=training)
                user_hidden = user_model([user_hidden, user_graph.edge_index, user_graph.edge_weight], training=training,
                                    cache=user_graph.cache)
                item_hidden = item_model([item_hidden, item_graph.edge_index, item_graph.edge_weight], training=training,
                                    cache=item_graph.cache)
                user_hidden = z_dropout(user_hidden, training=training)
                item_hidden = z_dropout(item_hidden, training=training)
                return user_hidden, item_hidden

        else:
            raise ValueError(f"Invalid method name: {self.args.method}")

        return forward

    @tf_utils.function
    def train(self, batch_user_indices, batch_item_indices, forward, optimizer):
        batch_negative_item_indices = tf.random.uniform(
            [tf.shape(batch_item_indices)[0], self.args.num_negs],
             0, self.num_items, dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            user_hidden, item_hidden = forward(training=True)

            embedded_users = tf.gather(user_hidden, batch_user_indices)
            embedded_items = tf.gather(item_hidden, batch_item_indices)
            embedded_neg_items = tf.gather(item_hidden, batch_negative_item_indices)

            query = tf.expand_dims(embedded_users, axis=-1)
            keys = tf.concat([
                tf.expand_dims(embedded_items, axis=1),
                embedded_neg_items
            ], axis=1)

            logits = tf.squeeze(keys @ query, axis=-1)
            mf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.zeros(tf.shape(logits)[0], dtype=tf.int32)
            )

            embedding_vars = [user_hidden, item_hidden]
            embedding_l2_losses = [tf.nn.l2_loss(var) for var in embedding_vars]
            embedding_l2_loss = tf.add_n(embedding_l2_losses)
            l2_loss = embedding_l2_loss * self.args.z_l2_coef
            loss = tf.reduce_sum(mf_losses) + l2_loss

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        return loss, mf_losses, l2_loss

    def run(self):
        time_stamp = int(time.time() * 1000)
        dataset_output_dir = os.path.join(self.args.output_dir, self.args.dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        result_path = os.path.join(dataset_output_dir, f"{self.args.method}_{self.args.dataset}_sp_{self.args.adj_drop_rate}_{time_stamp}.json")

        with open(result_path, "a", encoding="utf-8") as f:
            f.write(f"{json.dumps(vars(self.args))}\n")

        forward = self.get_model_and_forward()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)

        interval = 20
        time_spent = 0

        for epoch in range(self.args.num_epochs + 1):
            if epoch % interval == 0:
                user_hidden, item_hidden = forward(training=False)
                print(f"\nEvaluation before epoch {epoch} ......")
                mean_results_dict = evaluate_mean_global_metrics(self.test_user_item_dict, self.train_user_item_dict,
                                                                 user_hidden, item_hidden, k_list=[5, 10, 15, 20],
                                                                 metrics=["precision", "recall", "ndcg"])
                print(mean_results_dict)
                print()
                data = mean_results_dict.copy()
                data["epoch"] = epoch
                data["time"] = time_spent
                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(f"{json.dumps(data)}\n")

            step_losses = []
            step_mf_losses_list = []
            step_l2_losses = []

            start_time = time.time()
            for step, batch_edges in enumerate(
                    tf.data.Dataset.from_tensor_slices(self.train_user_item_edges).shuffle(
                        len(self.train_user_item_edges)).batch(self.args.batch_size)):
                batch_user_indices = batch_edges[:, 0]
                batch_item_indices = batch_edges[:, 1]
                loss, mf_losses, l2_loss = self.train(batch_user_indices, batch_item_indices, forward, optimizer)
                step_losses.append(loss.numpy())
                step_mf_losses_list.append(mf_losses.numpy())
                step_l2_losses.append(l2_loss.numpy())

            end_time = time.time()
            time_spent += (end_time - start_time)

            if optimizer.learning_rate.numpy() > 1e-5:
                optimizer.learning_rate.assign(optimizer.learning_rate * self.args.lr_decay)
                lr_status = f"update lr => {optimizer.learning_rate.numpy():.4f}"
            else:
                lr_status = f"current lr => {optimizer.learning_rate.numpy():.4f}"

            print(f"epoch = {epoch}\tloss = {np.mean(step_losses):.4f}\tmf_loss = {np.mean(np.concatenate(step_mf_losses_list, axis=0)):.4f}\tl2_loss = {np.mean(step_l2_losses):.4f}\t{lr_status}\tepoch_time = {end_time - start_time:.4f}s")


if __name__ == "__main__":
    args = parse_arguments()
    data_dict = setup_environment(args)
    task = CollaborativeFilteringTask(data_dict, args)
    task.run()
