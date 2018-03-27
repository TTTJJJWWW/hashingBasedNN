# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:10:08 2018
@email: dli@fiu.edu
@author: DeqiangLi
"""

import utils
from graphs import FFN_dae_lh_model

import tensorflow as tf
import numpy as np

import datetime
import os

flags = tf.flags
FLAGS = flags.FLAGS

#set parameters for trarning sequence CNN
flags.DEFINE_string("save_dir", "./save/jid/", 
                    "model parameters saved directory.")
flags.DEFINE_string("dataset", "./dataset/", 
                    "model parameters saved directory.")

flags.DEFINE_integer("batch_size", 128,
                     "size of batch for training, validation and test.")
flags.DEFINE_integer("adv_k", 64,
                     "number of adversarial examples in each batch_size.")
flags.DEFINE_integer("epoches", 30,
                     "the number of learing loops.")
flags.DEFINE_integer("output_dims", 2,
                     "the num of classificaton.")

flags.DEFINE_string("hidden_dims", "128,128",
                    "the neurons contained in each hidden layer.")
flags.DEFINE_integer("modified_num", 10,
                     "the maximum perturbations")
flags.DEFINE_integer("K", 128,
                     "the num of chose features per hashing")
flags.DEFINE_integer("L", 256,
                     "the num of hashing")
flags.DEFINE_bool("use_vanilla_embedding_params", True,
                  "training with vanilla model's embedding parameters. Keep sequence instinct.")

MAL_DATA_PATH = os.path.join(FLAGS.dataset, "malware.data")
BEN_DATA_PATH = os.path.join(FLAGS.dataset, "benign.data")
TRAIN_DATA_PATH = os.path.join(FLAGS.dataset, "train.data")
TRAIN_LABEL_PATH = os.path.join(FLAGS.dataset, "train.label")
TEST_DATA_PATH = os.path.join(FLAGS.dataset, "test.data")
TEST_LABEL_PATH = os.path.join(FLAGS.dataset, "test.label")


def random_noises_dae(batch_size, input_dim):
    mu = 0.0
    sigma = float(FLAGS.modified_num) / input_dim
    prob = np.clip(np.random.randn() * sigma + mu, a_min = 0., a_max = 1.)
    return np.random.binomial(1, prob, (batch_size, FLAGS.L, FLAGS.K))

def random_noises_vir(batch_size):
    return np.zeros((batch_size, FLAGS.L, FLAGS.K))

def train(sess, model, train_input, test_input, config):
    #training
    best_iter = 0
    best_acc = 0.
    MSG = "training start at {0}"
    print(MSG.format(str(datetime.datetime.now())))
    start_time = datetime.datetime.now()
    saver = tf.train.Saver()
    saver_dae = tf.train.Saver(model.share_vars)
    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)    
    train_bias = np.array([[10., 1.]], dtype = np.float32)
    test_bias = np.array([[1., 1]], dtype = np.float32)
    
    with sess.as_default():
        save_model_path = os.path.join(FLAGS.save_dir + "model.ckpt")
        n_epoches_dae = 80
        check_p = tf.train.checkpoint_exists(save_model_path)
        if not check_p:
            MSG = "starting denoising autoencoder training..."
            print(MSG)
            #n_epoches_dae = 20
            minimum_loss = 5. 
            for epoch in range(n_epoches_dae):
                train_input.reset_cursor()
                for mini_batch, trainX_batch, trainy_batch in train_input.next_batch():
                    train_dict = {model.x : trainX_batch,
                                  model.noises: random_noises_dae(trainX_batch.shape[0], trainX_batch.shape[1]),
                                  model.is_training: True
                              }
                    _, dae_train_loss = sess.run([model.dae_optimizer, model.dae_loss],
                                                    feed_dict= train_dict)
                
                    iterations = epoch * train_input.mini_batches + mini_batch + 1
                    if iterations % 100 == 0:
                        dae_test_loss = [sess.run(model.dae_loss, feed_dict = {model.x: testX_batch,
                                                                           model.noises: random_noises_vir(testX_batch.shape[0]),
                                                                           model.is_training: False}) \
                                     for [_, testX_batch, testy_batch] in test_input.next_batch()
                        ]
                        test_input.reset_cursor()
                        if np.mean(dae_test_loss) < minimum_loss:
                            #save model parameters
                            minimum_loss = np.mean(dae_test_loss)
                            save_model_path = os.path.join(FLAGS.save_dir + "model.ckpt")
                            saver.save(sess, save_model_path)
                        MSG = "The iteration is {0}, epoch {1}/{2}, minibatch {3}/{4}, with autoencoder training loss {5:.5} and test loss {6:.5}."
                        print(MSG.format(iterations, epoch + 1, n_epoches_dae, mini_batch + 1, train_input.mini_batches, dae_train_loss, np.mean(dae_test_loss)))
                
  
        MSG = "classifier training start at {0}"
        print(MSG.format(str(datetime.datetime.now())))
        start_time = datetime.datetime.now()
        for epoch in range(FLAGS.epoches):
            saver_dae.restore(sess, save_model_path)
            train_input.reset_cursor()
            for mini_batch, trainX_batch, trainy_batch in train_input.next_batch():
                train_dict = {model.x : trainX_batch,
                              model.noises:random_noises_vir(trainX_batch.shape[0]),
                              model.y : trainy_batch,
                              model.bias: train_bias,
                              model.is_training: True,
                              model.is_adv_training:False
                              }
                _, train_loss, train_acc = sess.run([model.update_clf, model.clf_loss, model.acc],
                                                    feed_dict= train_dict)
        
                iterations = epoch * train_input.mini_batches + mini_batch + 1
                
                if iterations % 100 == 0:
                    test_input.reset_cursor()
                    test_accs = [sess.run(model.acc, feed_dict = {model.x: testX_batch,
                                                                  model.noises:random_noises_vir(testX_batch.shape[0]),
                                                                  model.y: testy_batch,
                                                                  model.bias: test_bias,
                                                                  model.is_training: False}) \
                                 for [_, testX_batch, testy_batch] in test_input.next_batch()
                    ]
                    test_acc = np.mean(test_accs)
                    
                    MSG = "The iteration is {0}, epoch {1}/{2}, minibatch {3}/{4}, with training loss {5:.5}, training accuracy {6:.5} and test accuracy {7:.5}."
                    print(MSG.format(iterations, epoch + 1, FLAGS.epoches, mini_batch + 1, train_input.mini_batches, train_loss, train_acc, test_acc))
                    
                    if best_acc < test_acc:
                        best_iter = iterations
                        best_acc = test_acc
                        MSG = "\tThe best test accuracy is {0:.5}, which achieves at iteration {1}, wiht test accuracy {2:.5}"
                        print(MSG.format(best_acc, best_iter, test_acc))
                        #save model parameters
                        save_model_path = os.path.join(FLAGS.save_dir + "model.ckpt")
                        saver.save(sess, save_model_path)
                    
                    current_time = datetime.datetime.now()
                    MSG = "\tThis epoch training processing has cost {0} and the current time is {1}"
                    print(MSG.format(str(current_time - start_time), str(current_time))) 


def evaluate(sess, model, malX, benX):
    saver = tf.train.Saver()
    model_path = os.path.join(FLAGS.save_dir, "model.ckpt")
    #data_mal_ids = datay[:, 0] == 1.
    data_malX = malX
    data_maly = np.zeros((malX.shape[0], 2)).astype(np.float32)
    data_maly[:,0] = 1.
    data_benX = benX
    data_beny = np.zeros((benX.shape[0], 2)).astype(np.float32)
    data_beny[:, 1] = 1.
    test_bias = np.array([[1., 1]], dtype = np.float32)
    
    with sess.as_default():
        check_p = tf.train.checkpoint_exists(model_path)
        if check_p:
            saver.restore(sess, model_path)
        else:
            print("No saved parameters")
            
        def _evl(X, y):
            batches = X.shape[0] // FLAGS.batch_size + 1
            accs = []
            for mini_i in range(batches):
                start_i = mini_i * FLAGS.batch_size 
                end_i = FLAGS.batch_size + start_i
                if end_i > X.shape[0]:
                    end_i = X.shape[0]
                _feed_dict ={
                        model.x : X[start_i : end_i],
                        model.y : y[start_i : end_i],
                        model.noises: random_noises_vir(end_i - start_i),
                        model.bias: test_bias,
                        model.is_training: False,
                        model.is_adv_training : False
                        }
                acc = sess.run(model.acc, feed_dict = _feed_dict)
                accs.append(acc)
            return np.mean(accs)
        fnr = 1. - _evl(data_malX, data_maly)
        fpr = 1. - _evl(data_benX, data_beny)
        accuracy = float((1. - fnr) * data_malX.shape[0] + (1. - fpr) * data_benX.shape[0]) / (data_malX.shape[0] + data_benX.shape[0])
        
        MSG = "The accuracy of model is {0:.5}, with false negative rate {1:.5}, correponding false positive rate {2:.5}"
        print(MSG.format(accuracy, fnr, fpr))

def test(name = 'train'):
    if name == 'train':
        #read dataset
        trainX = utils.readdata_np(TRAIN_DATA_PATH)
        trainy = utils.readdata_np(TRAIN_LABEL_PATH)
        testX = utils.readdata_np(TEST_DATA_PATH)
        testy = utils.readdata_np(TEST_LABEL_PATH)
        train_input = utils.DataProducer(trainX, trainy, FLAGS.batch_size, FLAGS.epoches, "train")
        test_input = utils.DataProducer(testX, testy, FLAGS.batch_size, FLAGS.epoches, "test")
        
        input_dim = trainX.shape[1]
        _config = {
            'n_epoches':FLAGS.epoches,
            'batch_size':FLAGS.batch_size,
            'output_dim': FLAGS.output_dims,
            'input_dim' : input_dim,
            'hidden_dims' : list(map(int, FLAGS.hidden_dims.split(','))),
            'adv_k' : FLAGS.adv_k,
            'K' : FLAGS.K,  #recommned K =sqrt(input_dim)
            'L' : FLAGS.L
        }
        config = utils.ParamWrapper(_config)
        tf.reset_default_graph()
        model_mlp = FFN_dae_lh_model(config)
        tf.set_random_seed(1234)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        if name == 'train':
            train(sess, model_mlp, train_input, test_input, config)
    if name == 'evaluate':
        mal_data = utils.readdata_np(MAL_DATA_PATH)
        ben_data = utils.readdata_np(BEN_DATA_PATH)
        input_dim = mal_data.shape[1]
        _config = {
            'n_epoches':FLAGS.epoches,
            'batch_size':FLAGS.batch_size,
            'output_dim': FLAGS.output_dims,
            'input_dim' : input_dim,
            'hidden_dims' : list(map(int, FLAGS.hidden_dims.split(','))),
            'adv_k' : FLAGS.adv_k,
            'K' : FLAGS.K,  #recommned K =sqrt(input_dim)
            'L' : FLAGS.L
        }
        config = utils.ParamWrapper(_config)
        tf.reset_default_graph()
        model_mlp = FFN_dae_lh_model(config)
        tf.set_random_seed(1234)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        evaluate(sess, model_mlp, mal_data, ben_data)
    if name == 'experiments':
        ADV_SMPS_dir = './dataset/adv_smps/'
        ADV_SMPS10= os.path.join(ADV_SMPS_dir, 'adv_smp10.data')
        ADV_SMPS20 = os.path.join(ADV_SMPS_dir, 'adv_smp20.data')
        ADV_SMPS30 = os.path.join(ADV_SMPS_dir, 'adv_smp30.data')
        ADV_SMPS50 = os.path.join(ADV_SMPS_dir, 'adv_smp50.data')
        ADV_SMPS80 = os.path.join(ADV_SMPS_dir, 'adv_smp80.data')
        ADV_SMPS_OPT10 = os.path.join(ADV_SMPS_dir, 'adv_smps_opt10.data')
        ADV_SMPS_OPT20 = os.path.join(ADV_SMPS_dir, 'adv_smps_opt20.data')
        ADV_SMPS_OPT30 = os.path.join(ADV_SMPS_dir, 'adv_smps_opt30.data')
        CLEAN_SMPS_label_path = os.path.join(ADV_SMPS_dir, 'clean_smp.label')
        CLEAN_SMPS_path = os.path.join(ADV_SMPS_dir, 'clean_smp.data')
        adv_smps10 = utils.readdata_np(ADV_SMPS10)
        adv_smps20 = utils.readdata_np(ADV_SMPS20)
        adv_smps30 = utils.readdata_np(ADV_SMPS30)
        adv_smps50 = utils.readdata_np(ADV_SMPS50)
        adv_smps80 = utils.readdata_np(ADV_SMPS80)
        adv_smps_opt10 = utils.readdata_np(ADV_SMPS_OPT10)
        adv_smps_opt20 = utils.readdata_np(ADV_SMPS_OPT20)
        adv_smps_opt30 = utils.readdata_np(ADV_SMPS_OPT30)
        adv_smps0 = utils.readdata_np(CLEAN_SMPS_path)
        label = utils.readdata_np(CLEAN_SMPS_label_path)
        test_bias = np.array([[1., 1]], dtype = np.float32)
    
        input_dim = adv_smps0.shape[1]
        _config = {
            'n_epoches':FLAGS.epoches,
            'batch_size':FLAGS.batch_size,
            'output_dim': FLAGS.output_dims,
            'input_dim' : input_dim,
            'hidden_dims' : list(map(int, FLAGS.hidden_dims.split(','))),
            'adv_k' : FLAGS.adv_k,
            'K' : FLAGS.K,  #recommned K =sqrt(input_dim)
            'L' : FLAGS.L
        }
        config = utils.ParamWrapper(_config)
        tf.reset_default_graph()
        model_mlp = FFN_dae_lh_model(config)
        tf.set_random_seed(1234)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        model_path = os.path.join(FLAGS.save_dir, "model.ckpt")
        check_p = tf.train.checkpoint_exists(model_path)
        if check_p:
            saver.restore(sess, model_path)
        else:
            print("No saved parameters")
        with sess.as_default():   
            def _evl(X, y):
                batches = X.shape[0] // FLAGS.batch_size + 1
                accs = []
                for mini_i in range(batches):
                    start_i = mini_i * FLAGS.batch_size 
                    end_i = FLAGS.batch_size + start_i
                    if end_i > X.shape[0]:
                        end_i = X.shape[0]
                    _feed_dict ={
                            model_mlp.x : X[start_i : end_i],
                            model_mlp.y : y[start_i : end_i],
                            model_mlp.noises: random_noises_vir(end_i - start_i),
                            model_mlp.bias: test_bias,
                            model_mlp.is_training: False,
                            model_mlp.is_adv_training : False
                            }
                    acc = sess.run(model_mlp.acc, feed_dict = _feed_dict)
                    accs.append(acc)
                return np.mean(accs)
            acc1 = _evl(adv_smps0, label)
            acc10 = _evl(adv_smps10, label)
            acc20 = _evl(adv_smps20, label)
            acc30 = _evl(adv_smps30, label)
            acc50 = _evl(adv_smps50, label)
            acc80 = _evl(adv_smps80, label)
            acc10_opt = _evl(adv_smps_opt10, label)
            acc20_opt = _evl(adv_smps_opt20, label)
            acc30_opt = _evl(adv_smps_opt30, label)
        MSG = "The predict accuracy of clean data and FGS adversarial examples with perturbations [0, 10, 20, 30, 50, 80] is {0:.5}, {1:.5},{2:.5},{3:.5}, {4:.5},{5:.5} respectively."
        print(MSG.format(acc1, acc10, acc20, acc30, acc50, acc80))
        MSG = "The C.W. adversarial examples with perturbations [10, 20, 30] is {0:.5},{1:.5} and {2:.5}"
        print(MSG.format(acc10_opt, acc20_opt, acc30_opt))

def main(_):
    test(name = 'train')
    test(name = 'evaluate')
    #test(name='experiments')
    
if __name__ == '__main__':
    tf.app.run()
