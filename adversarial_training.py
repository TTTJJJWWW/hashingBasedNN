# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:10:08 2018

@author: DeqiangLi
"""

import utils
from graphs import FFN_model

import tensorflow as tf
import numpy as np

import datetime
import os

flags = tf.flags
FLAGS = flags.FLAGS

#set parameters for trarning sequence CNN
flags.DEFINE_string("save_dir_adv", "./save/adv_training/", 
                    "model parameters saved directory.")
flags.DEFINE_string("save_dir_std", "./save/standard_training/", 
                    "model parameters saved directory.")
flags.DEFINE_string("dataset", "./dataset/", 
                    "model parameters saved directory.")

flags.DEFINE_integer("batch_size", 128,
                     "size of batch for training, validation and test.")
flags.DEFINE_integer("adv_k", 64,
                     "number of adversarial examples in each batch_size.")
flags.DEFINE_integer("modified_num", 10,
                     "the maximum perturbations")
flags.DEFINE_integer("epoches", 30,
                     "the number of learing loops.")

flags.DEFINE_integer("output_dim", 2,
                     "the num of classificaton.")
flags.DEFINE_bool("use_vanilla_embedding_params", True,
                  "training with vanilla model's embedding parameters. Keep sequence instinct.")


MAL_DATA_PATH = os.path.join(FLAGS.dataset, "malware.data")
BEN_DATA_PATH = os.path.join(FLAGS.dataset, "benign.data")
TRAIN_DATA_PATH = os.path.join(FLAGS.dataset, "train.data")
TRAIN_LABEL_PATH = os.path.join(FLAGS.dataset, "train.label")
TEST_DATA_PATH = os.path.join(FLAGS.dataset, "test.data")
TEST_LABEL_PATH = os.path.join(FLAGS.dataset, "test.label")
ADV_DATA_PATH = os.path.join(FLAGS.dataset, "adv_smps.data")
ADV_LABEL_PATH = os.path.join(FLAGS.dataset, "adv_smps.label")

def adverarial_train(sess, model, train_input, test_input, test_adv_input, config):
    #training
    best_iter = 0
    best_acc = 0.
    MSG = "training start at {0}"
    print(MSG.format(str(datetime.datetime.now())))
    start_time = datetime.datetime.now()
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)    
    log = open(os.path.join(FLAGS.save_dir, 'adv_acc_log.txt'), 'w')
    with sess.as_default():    
        for epoch in range(FLAGS.epoches):
            train_input.reset_cursor()
            for mini_batch, trainX_batch, trainy_batch in train_input.next_batch():
                train_dict = {model.x : trainX_batch,
                              model.y : trainy_batch,
                              model.is_training: True,
                              model.is_adv_training : False
                              }
                _, train_loss, train_acc = sess.run([model.optimizer, model.loss, model.acc],
                                                    feed_dict= train_dict)
                
                test_adv_accs = [sess.run(model.acc, feed_dict = {model.x: testX_adv_batch,
                                                                  model.y: testy_adv_batch,
                                                                  model.is_training: False,
                                                                  model.is_adv_training: False
                                                                  }) \
                                for [_, testX_adv_batch, testy_adv_batch] in test_adv_input.next_batch()
                    ]
                test_adv_acc = np.mean(test_adv_accs)
                test_adv_input.reset_cursor()
                
                #we will log some scalars to observe the training performance
                iterations = epoch * train_input.mini_batches + mini_batch + 1
                buffer = str(test_adv_acc) + ' ' + str(iterations) + ' ' + str(epoch) + '\n'
                log.write(buffer)
                log.flush()
                
                #generating adversarial examples
                #select malware samples whose num does not exceed the FLAGS.adv_k
                
                trainX_adv_batch = trainX_batch.copy()
                mal_samples_idx = (trainy_batch[:,0] == 1.)
                mal_samples = trainX_adv_batch[mal_samples_idx]
                mal_labels = trainy_batch[mal_samples_idx]
                mal_samples_num = np.sum(mal_samples_idx)
                if mal_samples_num > config.adv_k:
                    mal_samples_num = config.adv_k
                mal_samples = trainX_adv_batch[mal_samples_idx.argsort()[-mal_samples_num:]]
                mal_labels = trainy_batch[mal_samples_idx.argsort()[-mal_samples_num:]]
                mal_adv_samples = model.generate_adv_smpls(sess, mal_samples, mal_labels, modified_num = FLAGS.modified_num)
                '''
                print(np.sum(mal_adv_samples - trainX_batch[mal_samples_idx.argsort()[-mal_samples_num:]]))
                train_dict = {model.x : mal_adv_samples,
                              model.y : mal_labels,
                              model.is_training: False,
                              model.is_adv_training : False
                              }
                tes_loss, t_acc, t_grad = sess.run([ model.loss, model.acc, model.grad],
                                                    feed_dict= train_dict)
                print(t_acc)
                '''
                trainX_adv_batch[mal_samples_idx.argsort()[-mal_samples_num:]] = mal_adv_samples
                                
                train_dict = {model.x : trainX_adv_batch,
                              model.y : trainy_batch,
                              model.is_training: True,
                              model.is_adv_training : True
                              }
                _, train_adv_loss, train_adv_acc = sess.run([model.optimizer, model.loss, model.acc],
                                                    feed_dict= train_dict)                
                
                if iterations % 100 == 0:
                    
                    test_accs = [sess.run(model.acc, feed_dict = {model.x: testX_batch,
                                                                   model.y: testy_batch,
                                                                   model.is_training: False,
                                                                   model.is_adv_training: False}) \
                                 for [_, testX_batch, testy_batch] in test_input.next_batch()
                    ]
                    test_acc = np.mean(test_accs)
                    test_input.reset_cursor()
                    MSG = "The iteration is {0}, epoch {1}/{2}, minibatch {3}/{4}, with training loss {5:.5}, training accuracy {6:.5} and validation accuracy {7:.5}."
                    print(MSG.format(iterations, epoch + 1, FLAGS.epoches, mini_batch + 1, train_input.mini_batches, train_loss, train_acc, test_acc))
                    #MSG = " \tThe adversarial training loss is {0:.5}, with adversarial examples training accuracy {1:.5} and test accuracy {2:.5}"
                    #print(MSG.format(train_adv_loss, train_adv_acc, test_adv_acc))
                    if best_acc < test_acc:
                        best_iter = iterations
                        best_acc = test_acc
                        MSG = "\tThe best test accuracy is {0:.5}, and achieve at iteration {1}"
                        print(MSG.format(test_acc, best_iter))
                        #save model parameters
                        save_model_path = os.path.join(FLAGS.save_dir + "model.ckpt")
                        saver.save(sess, save_model_path)
                    
                    current_time = datetime.datetime.now()
                    MSG = "\tThis epoch training processing has cost {0} and the current time is {1}"
                    print(MSG.format(str(current_time - start_time), str(current_time))) 
    log.close()
    
'''    
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
'''
def evaluate(sess, model, malX, benX, advX):
    saver = tf.train.Saver()
    model_path = os.path.join(FLAGS.save_dir_adv, "model.ckpt")
    #data_mal_ids = datay[:, 0] == 1.
    data_malX = malX
    data_maly = np.zeros((malX.shape[0], 2)).astype(np.float32)
    data_maly[:,0] = 1.
    data_benX = benX
    data_beny = np.zeros((benX.shape[0], 2)).astype(np.float32)
    data_beny[:, 1] = 1.
    data_advy = np.zeros((advX.shape[0], 2)).astype(np.float32)
    data_advy[:, 0] = 1.
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
                        model.is_training: False,
                        model.is_adv_training : False
                        }
                acc = sess.run(model.acc, feed_dict = _feed_dict)
                accs.append(acc)
            return np.mean(accs)
        fnr = 1. - _evl(data_malX, data_maly)
        fpr = 1. - _evl(data_benX, data_beny)
        adv_acc = _evl(advX, data_advy)
        accuracy = float((1. - fnr) * data_malX.shape[0] + (1. - fpr) * data_benX.shape[0]) / (data_malX.shape[0] + data_benX.shape[0])
        
        MSG = "The accuracy of model is {0:.5}, with adversersarial examples accuracy {1:.5}, false negative rate {2:.5}, correponding false positive rate {3:.5}"
        print(MSG.format(accuracy, adv_acc, fnr, fpr))

def test(name = 'train'):
    if name == 'train':
        #read dataset
        trainX = utils.readdata_np(TRAIN_DATA_PATH)
        trainy = utils.readdata_np(TRAIN_LABEL_PATH)
        testX = utils.readdata_np(TEST_DATA_PATH)
        testy = utils.readdata_np(TEST_LABEL_PATH)
        testX_adv = utils.readdata_np(ADV_DATA_PATH)
        #testy_adv = utils.readdata_np(ADV_LABEL_PATH)
        testy_adv = np.zeros((testX_adv.shape[0], 2)).astype(np.float32)
        testy_adv[:, 0] = 1.
        train_input = utils.DataProducer(trainX, trainy, FLAGS.batch_size, FLAGS.epoches, "train")
        test_input = utils.DataProducer(testX, testy, FLAGS.batch_size, FLAGS.epoches, "test")
        test_adv_input = utils.DataProducer(testX_adv, testy_adv, FLAGS.batch_size, FLAGS.epoches, "test_adv")
        
        input_dim = trainX.shape[1]
        _config = {
            'n_epoches':FLAGS.epoches,
            'batch_size':FLAGS.batch_size,
            'output_dim': FLAGS.output_dim,
            'input_dim' : input_dim,
            'adv_k' : FLAGS.adv_k
        }
        config = utils.ParamWrapper(_config)
        tf.reset_default_graph()
        model_mlp = FFN_model(config)
        tf.set_random_seed(1234)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        
        adverarial_train(sess, model_mlp, train_input, test_input, test_adv_input, config)
    if name == 'evaluate':
        mal_data = utils.readdata_np(MAL_DATA_PATH)
        ben_data = utils.readdata_np(BEN_DATA_PATH)
        testX_adv = utils.readdata_np(ADV_DATA_PATH)
        input_dim = mal_data.shape[1]
        _config = {
            'n_epoches':FLAGS.epoches,
            'batch_size':FLAGS.batch_size,
            'output_dim': FLAGS.output_dim,
            'input_dim' : input_dim,
            'adv_k' : FLAGS.adv_k
        }
        config = utils.ParamWrapper(_config)
        tf.reset_default_graph()
        model_mlp = FFN_model(config)
        tf.set_random_seed(1234)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        evaluate(sess, model_mlp, mal_data, ben_data, testX_adv)
    if name == 'experiments':
        ADV_SMPS_dir = './dataset/adv_smps/'
        ADV_SMPS10= os.path.join(ADV_SMPS_dir, 'adv_smp10.data')
        ADV_SMPS20 = os.path.join(ADV_SMPS_dir, 'adv_smp20.data')
        ADV_SMPS30 = os.path.join(ADV_SMPS_dir, 'adv_smp30.data')
        ADV_SMPS50 = os.path.join(ADV_SMPS_dir, 'adv_smp50.data')
        ADV_SMPS80 = os.path.join(ADV_SMPS_dir, 'adv_smp80.data')
        ADV_SMPS_OPT10 = os.path.join(ADV_SMPS_dir, 'adv_smps_opt.data')
        CLEAN_SMPS_label_path = os.path.join(ADV_SMPS_dir, 'clean_smp.label')
        CLEAN_SMPS_path = os.path.join(ADV_SMPS_dir, 'clean_smp.data')
        adv_smps10 = utils.readdata_np(ADV_SMPS10)
        adv_smps20 = utils.readdata_np(ADV_SMPS20)
        adv_smps30 = utils.readdata_np(ADV_SMPS30)
        adv_smps50 = utils.readdata_np(ADV_SMPS50)
        adv_smps80 = utils.readdata_np(ADV_SMPS80)
        adv_smps_opt = utils.readdata_np(ADV_SMPS_OPT10)
        adv_smps0 = utils.readdata_np(CLEAN_SMPS_path)
    
        label = utils.readdata_np(CLEAN_SMPS_label_path)
        input_dim = adv_smps0.shape[1]
        _config = {
                'n_epoches':FLAGS.epoches,
                'batch_size':FLAGS.batch_size,
                'output_dim': FLAGS.output_dim,
                'input_dim' : input_dim,
                'adv_k' : FLAGS.adv_k
            }
        config = utils.ParamWrapper(_config)
        tf.reset_default_graph()
        model_mlp = FFN_model(config)
        tf.set_random_seed(1234)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        model_path = os.path.join(FLAGS.save_dir_adv, "model.ckpt")
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
            acc10_opt = _evl(adv_smps_opt, label)
        MSG = "The predict accuracy of clean data and adversarial examples with perturbations [10, 20, 30, 50, 80, 10] is {0:.5}, {1:.5},{2:.5},{3:.5}, {4:.5},{5:.5}, {6:.5} respectively."
        print(MSG.format(acc1, acc10, acc20, acc30, acc50, acc80, acc10_opt))
        

def main(_):
    #test('train')
    #test('evaluate')
    test('experiments')
    
   
if __name__ == '__main__':
    tf.app.run()
