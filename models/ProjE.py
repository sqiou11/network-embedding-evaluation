import argparse
import math
import os.path
import timeit
from multiprocessing import JoinableQueue, Queue, Process

import numpy as np
import tensorflow as tf

class ProjE:
    @property
    def n_entity(self):
        return self.__n_entity

    @property
    def n_train(self):
        return self.__train_triple.shape[0]

    @property
    def trainable_variables(self):
        return self.__trainable

    @property
    def hr_t(self):
        return self.__hr_t

    @property
    def train_hr_t(self):
        return self.__train_hr_t

    @property
    def train_tr_h(self):
        return self.__train_tr_h

    @property
    def tr_h(self):
        return self.__tr_h

    @property
    def ent_embedding(self):
        return self.__ent_embedding

    @property
    def rel_embedding(self):
        return self.__rel_embedding

    def __init__(self, embed_dim=128, combination_method='simple', dropout=0.5, neg_weight=0.5):

        if combination_method.lower() not in ['simple', 'matrix']:
            raise NotImplementedError("ProjE does not support using %s as combination method." % combination_method)

        self.__combination_method = combination_method

        self.__embed_dim = embed_dim
        self.__initialized = False

        self.__trainable = list()
        self.__dropout = dropout
        self.__n_entity = 14541
        self.__n_relation = 237
        """
        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.__n_entity = len(f.readlines())

        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.__entity_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_entity_map = {v: k for k, v in self.__entity_id_map.items()}

        print("N_ENTITY: %d" % self.__n_entity)

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.__n_relation = len(f.readlines())

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.__relation_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_relation_map = {v: k for k, v in self.__entity_id_map.items()}

        print("N_RELATION: %d" % self.__n_relation)

        def load_triple(file_path):
            with open(file_path, 'r', encoding='utf-8') as f_triple:
                return np.asarray([[self.__entity_id_map[x.strip().split('\t')[0]],
                                    self.__entity_id_map[x.strip().split('\t')[1]],
                                    self.__relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
                                  dtype=np.int32)

        def gen_hr_t(triple_data):
            hr_t = dict()
            for h, t, r in triple_data:
                if h not in hr_t:
                    hr_t[h] = dict()
                if r not in hr_t[h]:
                    hr_t[h][r] = set()
                hr_t[h][r].add(t)

            return hr_t

        def gen_tr_h(triple_data):
            tr_h = dict()
            for h, t, r in triple_data:
                if t not in tr_h:
                    tr_h[t] = dict()
                if r not in tr_h[t]:
                    tr_h[t][r] = set()
                tr_h[t][r].add(h)
            return tr_h

        self.__train_triple = load_triple(os.path.join(data_dir, 'train.txt'))
        print("N_TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])

        self.__test_triple = load_triple(os.path.join(data_dir, 'test.txt'))
        print("N_TEST_TRIPLES: %d" % self.__test_triple.shape[0])

        self.__valid_triple = load_triple(os.path.join(data_dir, 'valid.txt'))
        print("N_VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

        self.__train_hr_t = gen_hr_t(self.__train_triple)
        self.__train_tr_h = gen_tr_h(self.__train_triple)
        self.__test_hr_t = gen_hr_t(self.__test_triple)
        self.__test_tr_h = gen_tr_h(self.__test_triple)

        self.__hr_t = gen_hr_t(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))
        self.__tr_h = gen_tr_h(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))
        """
        bound = 6 / math.sqrt(embed_dim)

        self.sess = tf.Session()
        with tf.device('/cpu'):
            self.__ent_embedding = tf.get_variable("ent_embedding", [self.__n_entity, embed_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound,
                                                                                                seed=345))
            self.__trainable.append(self.__ent_embedding)

            self.__rel_embedding = tf.get_variable("rel_embedding", [self.__n_relation, embed_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound,
                                                                                                seed=346))
            self.__trainable.append(self.__rel_embedding)

            if combination_method.lower() == 'simple':
                self.__hr_weighted_vector = tf.get_variable("simple_hr_combination_weights", [embed_dim * 2],
                                                            initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                        maxval=bound,
                                                                                                        seed=445))
                self.__tr_weighted_vector = tf.get_variable("simple_tr_combination_weights", [embed_dim * 2],
                                                            initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                        maxval=bound,
                                                                                                        seed=445))
                self.__trainable.append(self.__hr_weighted_vector)
                self.__trainable.append(self.__tr_weighted_vector)
                self.__hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                                initializer=tf.zeros([embed_dim]))
                self.__tr_combination_bias = tf.get_variable("combination_bias_tr",
                                                                initializer=tf.zeros([embed_dim]))

                self.__trainable.append(self.__hr_combination_bias)
                self.__trainable.append(self.__tr_combination_bias)

            else:
                self.__hr_combination_matrix = tf.get_variable("matrix_hr_combination_layer",
                                                                [embed_dim * 2, embed_dim],
                                                                initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                            maxval=bound,
                                                                                                            seed=555))
                self.__tr_combination_matrix = tf.get_variable("matrix_tr_combination_layer",
                                                                [embed_dim * 2, embed_dim],
                                                                initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                            maxval=bound,
                                                                                                            seed=555))
                self.__trainable.append(self.__hr_combination_matrix)
                self.__trainable.append(self.__tr_combination_matrix)
                self.__hr_combination_bias = tf.get_variable("combination_bias_hr",
                                                                initializer=tf.zeros([embed_dim]))
                self.__tr_combination_bias = tf.get_variable("combination_bias_tr",
                                                                initializer=tf.zeros([embed_dim]))

                self.__trainable.append(self.__hr_combination_bias)
                self.__trainable.append(self.__tr_combination_bias)
                
            # load model from file
            tf.train.Saver().restore(self.sess, './embeddings/proje/ProjE_DEFAULT_39.ckpt')

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor * tf.abs(weights), 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp * tf.abs(weights), 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * tf.abs(weights)  # all ignored elements will have a prob of 0.

    def predict(self, head, tail, rel, pred_tail=True, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()

            #inputs = tf.placeholder(tf.int32, [None, 3])
            #h = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 0])
            #t = tf.nn.embedding_lookup(self.__ent_embedding, inputs[:, 1])
            #r = tf.nn.embedding_lookup(self.__rel_embedding, inputs[:, 2])
            h = tf.nn.embedding_lookup(self.__ent_embedding, np.asarray([head]))
            t = tf.nn.embedding_lookup(self.__ent_embedding, np.asarray([tail]))
            r = tf.nn.embedding_lookup(self.__rel_embedding, np.asarray([rel]))
            ent_mat = tf.transpose(self.__ent_embedding)

            if self.__combination_method.lower() == 'simple':

                # predict tails
                if pred_tail:
                    hr = h * self.__hr_weighted_vector[:self.__embed_dim] + r * self.__hr_weighted_vector[
                                                                                self.__embed_dim:]
                    hrt_res = tf.matmul(tf.tanh(hr + self.__hr_combination_bias), ent_mat)
                    #print(hrt_res)
                    #_, tail_ids = tf.nn.top_k(hrt_res, k=self.__n_entity)
                    tail_pred = self.sess.run(hrt_res)
                    return tail_pred.tolist()[0]

                # predict heads
                else:
                    tr = t * self.__tr_weighted_vector[:self.__embed_dim] + r * self.__tr_weighted_vector[self.__embed_dim:]

                    trh_res = tf.matmul(tf.tanh(tr + self.__tr_combination_bias), ent_mat)
                    #_, head_ids = tf.nn.top_k(trh_res, k=self.__n_entity)
                    head_pred = self.sess.run(trh_res)
                    return head_pred.tolist()[0]

            else:
                if pred_tail:
                    hr = tf.matmul(tf.concat(1, [h, r]), self.__hr_combination_matrix)
                    hrt_res = (tf.matmul(tf.tanh(hr + self.__hr_combination_bias), ent_mat))
                    #_, tail_ids = tf.nn.top_k(hrt_res, k=self.__n_entity)
                    tail_pred = self.sess.run(hrt_res)
                    return tail_pred.tolist()[0]
                else:
                    tr = tf.matmul(tf.concat(1, [t, r]), self.__tr_combination_matrix)
                    trh_res = (tf.matmul(tf.tanh(tr + self.__tr_combination_bias), ent_mat))
                    #_, head_ids = tf.nn.top_k(trh_res, k=self.__n_entity)
                    head_pred = self.sess.run(trh_res)
                    return head_pred.tolist()[0]

            #head_pred, tail_pred = self.sess.run([head_ids, tail_ids], {inputs: [[6180, 2861, 148]]})
            #for val in head_pred:
            #print(head_pred.tolist()[0].index(6180), tail_pred.tolist()[0].index(2861))
            #print (self.sess.run(tail_ids))
            #return head_ids, tail_ids