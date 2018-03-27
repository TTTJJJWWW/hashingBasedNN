# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:29:13 2018
@email: dli@fiu.edu
@author: DeqiangLi
"""
import datetime
import os

import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops, tensor_array_ops
import numpy as np
import math

import utils
from learning_hashing_by_RF import RF_hashing, F_hashing
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.001,
                   "learning rate.")
flags.DEFINE_float("keep_prob", 0.6,
                   "the probability of drop data after fully concat layer.")
flags.DEFINE_float("param_lambda", 0.5,
                   "The parameters for adversarial training.")

def FFN_graph(_x, is_training, initializer, hidden_units = [128, 128], output_dim = 2, keep_probs = 0.5, reuse = False):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9
                         , 'updates_collections': None}
    fc1 = slim.fully_connected(_x, hidden_units[0], 
                               activation_fn = tf.nn.relu,
                               weights_initializer = initializer,
                               normalizer_fn = slim.batch_norm,
                               normalizer_params = batch_norm_params,
                               scope = 'fc1',
                               reuse = reuse
                              )
    fc2 = slim.fully_connected(fc1, hidden_units[1],
                               activation_fn = tf.nn.relu,
                               weights_initializer = initializer,
                               normalizer_fn = slim.batch_norm,
                               normalizer_params = batch_norm_params,
                               scope = 'fc2',
                               reuse = reuse
                              )
    dropout2 = slim.dropout(fc2, keep_prob = keep_probs, is_training = is_training, scope = 'dropout2')
    fc3 = slim.fully_connected(dropout2, output_dim,
                               activation_fn = None,
                               scope = 'fc3',
                               reuse = reuse
                              )
    return fc3

def FNN_Hashing_graph(_x,  is_training, initializer, hashing_table, hidden_units = [128, 128], output_dim = 2, keep_probs = 0.4, reuse = False): 
    x_hashing = hashing_layer(_x, hashing_table) #L * batch_size * K
    #dropout
    x_hashing = tf.transpose(x_hashing, (1, 0, 2))
    x_hashing = tf.cond(is_training, lambda : dropout_channels(x_hashing, keep_prob= FLAGS.keep_prob, L = hashing_table.shape[0]), lambda: x_hashing)
    x_hashing = dropout_channels(x_hashing, keep_prob= FLAGS.keep_prob, L = hashing_table.shape[0])
    with tf.variable_scope("Hashing_layer"):
        L = int(hashing_table.shape[0])
        K = int(hashing_table.shape[1])
        W1 = tf.Variable(tf.random_normal([L, K, hidden_units[0]], stddev = 0.01), name = 'W1')
        b1 = tf.Variable(tf.zeros([L, hidden_units[0]]), name = 'b1')
    
    hl1 = tf.nn.relu(tf.transpose(tf.matmul(tf.transpose(x_hashing,(1, 0, 2)), W1), (1, 0, 2)) + b1)
   
    hl1_out = tf.contrib.layers.flatten(hl1) #=>(batchsize, L, hidden_dim)=>(batch_size, fc_length)
    
    dropout1 = slim.dropout(hl1_out, keep_prob = keep_probs, is_training = is_training, scope = 'dropout1')
    
    fc1 = slim.fully_connected(dropout1, hidden_units[1], 
                               activation_fn = tf.nn.relu,
                               weights_initializer = initializer,
                               scope = 'fc1',
                               reuse = reuse
                              )
    
    fc2 = slim.fully_connected(fc1, hidden_units[1],
                               activation_fn = tf.nn.relu,
                               weights_initializer = initializer,
                               scope = 'fc2',
                               reuse = reuse
                              )
    
    dropout2 = slim.dropout(fc2, keep_prob = keep_probs, is_training = is_training, scope = 'dropout2')
    fc3 = slim.fully_connected(dropout2, output_dim,
                               activation_fn = None,
                               scope = 'fc3',
                               reuse = reuse
                              )
    return fc3

def FNN_dae_graph(_x, is_training, initializer, hashing_table, noises,
                  hidden_units = [128, 128], output_dim = 2,
                  keep_probs = 0.6, reuse = False):
    x_hashing = hashing_layer(_x, hashing_table) #L * batch_size * K
    x_hashing = tf.transpose(x_hashing, (1, 0, 2))
    #inject noises
    x_hashing_noise = tf.cond(is_training, lambda: tf.clip_by_value(x_hashing + noises, clip_value_min= 0., clip_value_max= 1.), lambda: x_hashing)
    #dropout 01
    x_hashing_out = tf.cond(is_training, lambda: dropout_channels(x_hashing_noise, keep_prob = keep_probs, L = hashing_table.shape[0]), lambda: x_hashing_noise)
    with tf.variable_scope("Hashing_layer_share", reuse = reuse):
        L = int(hashing_table.shape[0])
        K = int(hashing_table.shape[1])
        W1 = tf.Variable(tf.random_normal([L,K,hidden_units[0]], stddev = 0.01), name = 'W1')
        b1 = tf.Variable(tf.zeros([L,hidden_units[0]]), name = 'b1')
        
        hl1 = tf.nn.relu(tf.transpose(tf.matmul(tf.transpose(x_hashing_out, (1, 0, 2)), W1), (1, 0, 2)) + b1)
        hl1_flatten = tf.contrib.layers.flatten(hl1)
        #dropout 02
        hl1_out = slim.dropout(hl1_flatten, keep_prob= keep_probs, is_training = is_training, scope = 'dropout_flatten1')
        #fc1
        
        encoding01 = slim.fully_connected(hl1_out, hidden_units[0],
                                          activation_fn = tf.nn.relu,
                                          weights_initializer = initializer,
                                          scope = 'encoding01'
                                         )
        
        encoding01_out = slim.dropout(encoding01, keep_prob = keep_probs, is_training = is_training, scope = 'dropout_encoding1')
        
    with tf.variable_scope("Decoder"):
        decoding02 = slim.fully_connected(encoding01_out, L * K,
                                          activation_fn = None,
                                          scope = 'decoding02'
                                         )
        
        dae_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_hashing, logits= tf.reshape(decoding02, (-1, L, K))))
        #dae_loss = 0.5 * tf.reduce_mean(tf.pow(x_hashing - tf.reshape(tf.nn.relu(decoding02), (-1, L, K)), 2.0)) #the performance matches with cross_entropy
    with tf.variable_scope("Feed_Foward_Nets"):
        #fc2
        fc02 = slim.fully_connected(encoding01, hidden_units[1],
                                    activation_fn = tf.nn.relu,
                                    weights_initializer = initializer,
                                    scope = 'fc2'
                                   )
        fc02_out = slim.dropout(fc02, keep_prob= keep_probs, is_training = is_training, scope = 'dropout_fc02')
        
        fc03 = slim.fully_connected(fc02_out, output_dim,
                                   activation_fn = None,
                                   scope = 'fc3'
                                  )
    return dae_loss, fc03

def FNN_representation_graph(_x, is_training, initializer, input_dim, output_dim, hidden_units = [128, 128], 
                             keep_prob = 0.6,
                             acti_fn = tf.nn.relu,
                             reuse = False):
    _x = tf.cond(is_training, lambda: dropout_channels(_x, 1.0, L = int(_x.get_shape()[1])), lambda: _x)
    with tf.variable_scope("learning_represent"):
        W1 = tf.Variable(tf.random_normal([input_dim[0], input_dim[1], hidden_units[0]], stddev = 0.01), name = 'W1') #L,K,hidden_units
        b1 = tf.Variable(tf.zeros([input_dim[0], hidden_units[0]]), name = "b1")
    _x = tf.transpose(_x, (1, 0, 2))
    ln_rf = acti_fn(tf.transpose(tf.matmul(_x, W1), (1, 0, 2)) + b1)
    
    ln_rf_flatten = tf.contrib.layers.flatten(ln_rf)
    dropout1 = slim.dropout(ln_rf_flatten, keep_prob = 1., is_training = is_training, scope = 'dropout1') #no droputout1
    
    fc1 = slim.fully_connected(dropout1, hidden_units[1], 
                               activation_fn = acti_fn,
                               weights_initializer = initializer,
                               scope = 'fc1',
                               reuse = reuse
                              )
    
    
    fc2 = slim.fully_connected(fc1, hidden_units[1],
                               activation_fn = acti_fn,
                               weights_initializer = initializer,
                               scope = 'fc2',
                               reuse = reuse
                              )
    
    dropout2 = slim.dropout(fc2, keep_prob = keep_prob, is_training = is_training, scope = 'dropout2')
    
    fc3 = slim.fully_connected(dropout2, output_dim,
                               activation_fn = None,
                               scope = 'fc3',
                               reuse = reuse
                              )
    
    return fc3

def FNN_representation_dae_graph(_x, is_training, initializer, noises, input_dim, output_dim, hidden_units = [128, 128], 
                             keep_prob = 0.6,
                             acti_fn = tf.nn.relu,
                             reuse = False):
    L = int(_x.get_shape()[1])
    K_ext = int(_x.get_shape()[2])
    #injecting noises
    _x_noise = tf.cond(is_training, lambda: tf.clip_by_value(_x + noises, clip_value_min= 0., clip_value_max= 1.), lambda: _x)
    #channel level dropout
    _x_out = tf.cond(is_training, lambda: dropout_channels(_x_noise,keep_prob=1.,  L = L), lambda: _x_noise)
    with tf.variable_scope("learning_represent_share"):
        W1 = tf.Variable(tf.random_normal([input_dim[0], input_dim[1], hidden_units[0]], stddev = 0.01), name = 'W1') #L,K,hidden_units
        b1 = tf.Variable(tf.zeros([input_dim[0], hidden_units[0]]), name = "b1")
        _x_out = tf.transpose(_x_out, (1, 0, 2))
        ln_rf = acti_fn(tf.transpose(tf.matmul(_x_out, W1), (1, 0, 2)) + b1)
    
        ln_rf_flatten = tf.contrib.layers.flatten(ln_rf)
        ln_rf_out = slim.dropout(ln_rf_flatten, keep_prob = keep_prob, is_training = is_training, scope = 'ln_rf_out')
        #encoding (i.e. fc1)
        fc1 = slim.fully_connected(ln_rf_out, hidden_units[0],
                                   activation_fn = acti_fn,
                                   weights_initializer = initializer,
                                   scope = 'fc1',
                                   reuse = reuse
                                  )
        fc1_out = slim.dropout(fc1, keep_prob = keep_prob, is_training = is_training, scope = 'fc1_out')
    with tf.variable_scope("Decoder_lfh"):
        decoding01 = slim.fully_connected(fc1_out, L * K_ext,
                                          activation_fn = None,
                                          scope = 'decoding01'
                                         )
        dae_loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=_x, logits= tf.reshape(decoding01, (-1, L, K_ext))))
    
    with tf.variable_scope("FNN_lfh"):
        fc2 = slim.fully_connected(fc1, hidden_units[1],
                                   activation_fn = acti_fn,
                                   weights_initializer = initializer,
                                   scope = 'fc2',
                                   reuse = reuse
                                  )
    
        fc2_out = slim.dropout(fc2, keep_prob = keep_prob, is_training = is_training, scope = 'fc2_out')
    
        fc3 = slim.fully_connected(fc2_out, output_dim,
                                   activation_fn = None,
                                   scope = 'fc3',
                                   reuse = reuse
                                  )
    
    return dae_loss, fc3


def hashing_layer(input_tensor, table):
    L = table.shape[0]
    x_hashing_code = tf.TensorArray(dtype = tf.float32, size = L, dynamic_size = False, infer_shape = True, name = "H_C")
    def _bdy(i, x_hashing_code):
        hashing = tf.gather(input_tensor, table[i], axis = 1)
        x_hashing_code = x_hashing_code.write(i, hashing)
        return i + 1, x_hashing_code
    _, _x_hashing = tf.while_loop(
            cond = lambda i, _1 : i < L,
            body = _bdy,
            loop_vars= (tf.constant(0, tf.int32), x_hashing_code)
            )
    return _x_hashing.stack() #L * batch_size * K

def dropout_channels(input_tensor, keep_prob = 0.4, L = 50):
    n_channles = int(L)
    random_tensor = tf.floor(
                    tf.random_uniform(shape = [n_channles]) + tf.convert_to_tensor(keep_prob, dtype = tf.float32, name = "keep_prob"))
    drop = tf.transpose(input_tensor, (0, 2, 1)) * random_tensor
    return tf.transpose(drop, (0, 2, 1))

class FFN_model(object):
    def __init__(self,config):
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.batch_size = config.batch_size
        self.adv_k = config.adv_k
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.param_lambda = FLAGS.param_lambda
        self.x = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim], name = 'X')
        self.y = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'Y')
        self.is_training = tf.placeholder(tf.bool)
        self.is_adv_training = tf.placeholder(tf.bool)
        self.initizlizer = tf.random_normal_initializer(0, 0.01)
        self.mlp = FFN_graph(self.x, self.is_training, self.initizlizer, keep_probs= self.keep_prob, output_dim=self.output_dim)
        self.softmax_out = tf.nn.softmax(self.mlp)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp))
        softmax_logists = tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp)
        loss_adv = (1. / (self.batch_size - self.adv_k + self.param_lambda * self.adv_k)) *\
                        (self.param_lambda * tf.reduce_sum(softmax_logists[:self.adv_k]) + tf.reduce_sum(softmax_logists[self.adv_k:]))
        self.loss = tf.cond(self.is_adv_training, lambda : loss_adv, lambda : loss)
        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.mlp, axis = 1), tf.argmax(self.y, axis = 1)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss) 
    
    def fgsm_graph(self):
        pred_max = tf.reduce_max(self.mlp, axis = 1, keep_dims= True)
        pred_y = tf.to_float(tf.equal(pred_max, self.mlp))
        pred_y = tf.stop_gradient(pred_y)
        pred_y = pred_y / tf.reduce_sum(pred_y, 1, keep_dims= True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= pred_y, logits= self.mlp))
        grad, = tf.gradients(loss, self.x)
        return grad
        
    def generate_adv_smpls(self, sess, mal_samples, mal_labels, modified_num = 20):
        self.grad = self.fgsm_graph()
        with sess.as_default():
            feed_dict = {
                self.x : mal_samples,
                self.y : mal_labels,
                self.is_training: False,
                self.is_adv_training:False
                }
            grads = sess.run(self.grad, feed_dict = feed_dict)
            pos_invalid = (mal_samples == 1.)
            grads[pos_invalid] = np.min(grads)
            grads = np.clip(grads, a_min = 0., a_max = 12345.)
            pos_perturbed = np.argpartition(grads, np.argmin(grads, axis = 0))[:,-modified_num:]
            for i, mal in enumerate(mal_samples):
                mal[pos_perturbed[i]] = 1.
                   
            return mal_samples
        
#============================================== Local hashing defense - Index function =====================================================        
        
class FFN_lh_model(object):
    def __init__(self,config):
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.batch_size = config.batch_size
        self.adv_k = config.adv_k
        self.K = config.K
        self.L = config.L
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.param_lambda = FLAGS.param_lambda
        self.x = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim], name = 'X')
        self.y = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'Y')
        self.bias = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'BIAS')
        self.is_training = tf.placeholder(tf.bool)
        self.is_adv_training = tf.placeholder(tf.bool)
        self.table = tf.constant(self._hashing_table(replacement = True), dtype = tf.int32)
        self.initizlizer = tf.random_normal_initializer(0, 0.01)
        self.mlp = FNN_Hashing_graph(self.x, self.is_training, self.initizlizer, self.table, hidden_units = config.hidden_dims, keep_probs= self.keep_prob, output_dim=self.output_dim)
        self.softmax_out = tf.nn.softmax(self.mlp)
        self.mlp_bias = self.mlp / self.bias
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp_bias))
        softmax_logists = tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp_bias)
        loss_adv = (1. / (self.batch_size - self.adv_k + self.param_lambda * self.adv_k)) *\
                        (self.param_lambda * tf.reduce_sum(softmax_logists[:self.adv_k]) + tf.reduce_sum(softmax_logists[self.adv_k:]))
        self.loss = tf.cond(self.is_adv_training, lambda : loss_adv, lambda : loss)
        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.softmax_out, axis = 1), tf.argmax(self.y, axis = 1)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss) 
    
    def _hashing_table(self, replacement = True):
        if replacement:
            np.random.seed(2345)
            return np.random.randint(0, self.input_dim, size = (self.L, self.K))
        else:
            import random
            random.seed(2345)
            return np.array([random.sample(range(0, self.input_dim), self.K) for l in range(self.L)],dtype = np.int32)
    
    def _fgsm_graph(self):
        pred_max = tf.reduce_max(self.mlp, axis = 1, keep_dims= True)
        pred_y = tf.to_float(tf.equal(pred_max, self.mlp))
        pred_y = tf.stop_gradient(pred_y)
        pred_y = pred_y / tf.reduce_sum(pred_y, 1, keep_dims= True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= pred_y, logits= self.mlp))
        grad, = tf.gradients(loss, self.x)
        return grad
        
    def generate_adv_smpls(self, sess, mal_samples, mal_labels, modified_num = 10):
        self.grad = self._fgsm_graph()
        with sess.as_default():
            feed_dict = {
                self.x : mal_samples,
                self.y : mal_labels,
                self.is_training: False,
                self.is_adv_training:False
                }
            grads = sess.run(self.grad, feed_dict = feed_dict)
            pos_invalid = (mal_samples == 1.)
            grads[pos_invalid] = np.min(grads)
            grads = np.clip(grads, a_min = 0., a_max = 12345.)
            pos_perturbed = np.argpartition(grads, np.argmin(grads, axis = 0))[:,-modified_num:]
            for i, mal in enumerate(mal_samples):
                mal[pos_perturbed[i]] = 1.
                   
            return mal_samples       
        
#==============================================Local hashing defense-Index function ====================================================        
class FFN_dae_lh_model(object):
    def __init__(self,config):
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.batch_size = config.batch_size
        self.adv_k = config.adv_k
        self.K = config.K
        self.L = config.L
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.param_lambda = FLAGS.param_lambda
        self.x = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim], name = 'X')
        self.y = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'Y')
        self.bias = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'BIAS')
        self.is_training = tf.placeholder(tf.bool)
        self.is_adv_training = tf.placeholder(tf.bool)
        self.noises = tf.placeholder(dtype = tf.float32, shape = [None, self.L, self.K], name = "NOISE")
        self.table = tf.constant(self._hashing_table(replacement = True), dtype = tf.int32)
        self.initizlizer = tf.random_normal_initializer(0, 0.01)
        self.dae_loss, self.mlp = FNN_dae_graph(self.x, self.is_training, self.initizlizer, self.table, self.noises, hidden_units = config.hidden_dims, keep_probs= self.keep_prob, output_dim=self.output_dim)
        #==============================================denosing autoencoder============================
        self.dae_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dae_loss)
        self.share_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Hashing_layer_share")
        #==============================================feed fowared nets===============================
        self.softmax_out = tf.nn.softmax(self.mlp)
        self.mlp_bias = self.mlp / self.bias
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp_bias))
        softmax_logists = tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp_bias)
        loss_adv = (1. / (self.batch_size - self.adv_k + self.param_lambda * self.adv_k)) *\
                        (self.param_lambda * tf.reduce_sum(softmax_logists[:self.adv_k]) + tf.reduce_sum(softmax_logists[self.adv_k:]))
        self.clf_loss = tf.cond(self.is_adv_training, lambda : loss_adv, lambda : loss)
        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.softmax_out, axis = 1), tf.argmax(self.y, axis = 1)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
	
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= "Feed_Foward_Nets")
        self.gradients_and_vars = self.optimizer.compute_gradients(self.clf_loss, self.vars)
        self.update_clf = self.optimizer.apply_gradients(self.gradients_and_vars)
        #training together, with improving the accuracy but decreasing resistance
        self.loss = self.dae_loss + self.clf_loss
        self.update = self.optimizer.minimize(self.loss)

    def _hashing_table(self, replacement = True):
        if replacement:
            np.random.seed(2345)
            return np.random.randint(0, self.input_dim, size = (self.L, self.K))
        else:
            import random
            random.seed(2345)
            return np.array([random.sample(range(0, self.input_dim), self.K) for l in range(self.L)],dtype = np.int32)   


#==============================================Local hashing defense-Local forest function ====================================================              
class lhrf_model(object):
    def __init__(self, config, trainX, trainy, is_load_param = False):
        self.input_dim = config.input_dim
        self.K = config.K
        self.L = config.L
        self.tables = self._hashing_table_f(True)
        self.is_load = is_load_param

        self.learning_hashing_by_rf = F_hashing(trainX, trainy, self.tables, k = self.K, l = self.L)
        #the alternative choice is 
        '''
        self.tables = self._hashing_table_rf(True)
        self.learning_hashing_by_rf = RF_hashing(trainX, trainy, self.tables, k = self.K, l = self.L)
        '''
        self._learning_map(os.path.join(config.save_dir, 'rf.represent'), self.is_load)
    
        
    def _hashing_table_f(self, replacement = True):
        m = int(math.sqrt(self.input_dim))
        if replacement:
            np.random.seed(2345)
            return np.random.randint(0, self.input_dim, size = (self.L, self.K, m))
        else:
            import random
            random.seed(2345)
            return np.array([[random.sample(range(0, self.input_dim), m) for k in range(self.K)] \
                             for l in range(self.L)],dtype = np.int32)
    
    #using random forest substitutes the forest algorithms
    def _hashing_table_rf(self, replacement = True):
        m = int(math.sqrt(self.input_dim)) * self.K // 2
        if replacement:
            np.random.seed(2345)
            return np.random.randint(0, self.input_dim, size = (self.L, m))
        else:
            import random
            random.seed(2345)
            return np.array([random.sample(range(0, self.input_dim), m) for l in range(self.L)],dtype = np.int32)
    def _learning_map(self, save_path, is_load = False):
        if os.path.exists(save_path) and is_load:
            self.learning_hashing_by_rf.load_params(save_path)
        else:
            self.learning_hashing_by_rf.learning_map()
            self.learning_hashing_by_rf.dump_params(save_path)
        #self.learning_hashing_by_rf.learning_map()
        #self.learning_hashing_by_rf.dump_params(save_path)
        
#==============================================Local hashing defense-Feed_Forward model====================================================                      
class FFN_lhrf_model(object):
    def __init__(self, config):
        self.input_dim = config.input_dim  #len(input_dim) == 2
        self.output_dim = config.output_dim
        self.batch_size = config.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.x = tf.placeholder(tf.float32, [None,self.input_dim[0], self.input_dim[1]], name = 'X')
        self.y = tf.placeholder(tf.float32, [None, self.output_dim], name = 'Y')
        self.bias = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'BIAS')
        self.is_training = tf.placeholder(tf.bool)    
        
        self.initizlizer = tf.random_normal_initializer(0, 0.01)
        self.mlp = FNN_representation_graph(self.x, self.is_training, self.initizlizer, 
                                            keep_prob = self.keep_prob, 
                                            input_dim= self.input_dim,
                                            output_dim=self.output_dim,
                                            hidden_units= config.hidden_dims
                                            )
        
        self.softmax_out = tf.nn.softmax(self.mlp)
        self.mlp_bias = self.mlp / self.bias
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp_bias))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.softmax_out, axis = 1), tf.argmax(self.y, axis = 1)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss) 

#==============================================Local hashing defense-Feed_Forward nets and DAE model====================================================
class FFN_lhrf_dae_model(object):
    def __init__(self, config):
        self.input_dim = config.input_dim  #len(input_dim) == 2
        self.output_dim = config.output_dim
        self.batch_size = config.batch_size
        self.K_ext = config.input_dim[1]
        self.L = config.input_dim[0]
        self.learning_rate = FLAGS.learning_rate
        self.keep_prob = FLAGS.keep_prob
        self.x = tf.placeholder(tf.float32, [None,self.input_dim[0], self.input_dim[1]], name = 'X')
        self.y = tf.placeholder(tf.float32, [None, self.output_dim], name = 'Y')
        self.bias = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'BIAS')
        self.is_training = tf.placeholder(tf.bool)    
        self.noises = tf.placeholder(dtype = tf.float32, shape = [None, self.L, self.K_ext], name = "NOISE")
        self.initizlizer = tf.random_normal_initializer(0, 0.01)
        self.dae_loss, self.mlp = FNN_representation_dae_graph(self.x, self.is_training, self.initizlizer, 
                                                self.noises,
                                                keep_prob = self.keep_prob, 
                                                input_dim= self.input_dim,
                                                output_dim=self.output_dim,
                                                hidden_units= config.hidden_dims
                                               )
        #==============================================denosing autoencoder============================
        self.dae_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dae_loss)
        self.share_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='learning_represent_share')
        #==============================================feed fowared nets===============================
        self.softmax_out = tf.nn.softmax(self.mlp)
        self.mlp_bias = self.mlp / self.bias
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= self.y, logits= self.mlp_bias))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.softmax_out, axis = 1), tf.argmax(self.y, axis = 1)), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'FNN_lfh')
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.vars)
        self.update = self.optimizer.apply_gradients(self.grads_and_vars)
