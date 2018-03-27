#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:11:16 2018
Robust Support vector machine 
Robust logitic regression
@author: dli
"""
import os

import math
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib

import utils
from learning_hashing_by_RF import RF_hashing
from learning_hashing_by_RF import F_hashing

#hyper parameters
K = 4
L = 8
save_dir = './save/'
dataset = './dataset/'

#data path
MAL_DATA_PATH = os.path.join(dataset, "malware.data")
BEN_DATA_PATH = os.path.join(dataset, "benign.data")
TRAIN_DATA_PATH = os.path.join(dataset, "train.data")
TRAIN_LABEL_PATH = os.path.join(dataset, "train.label")
TEST_DATA_PATH = os.path.join(dataset, "test.data")
TEST_LABEL_PATH = os.path.join(dataset, "test.label")
ADV_DATA_PATH = os.path.join(dataset, "adv_smps.data")
ADV_LABEL_PATH = os.path.join(dataset, "adv_smps.label")

class lr_rpst(object):
    def __init__(self, config):
        self.K = config.K
        self.L = config.L
        self.max_depth = config.max_depth
        self.input_dim = config.input_dim
        self.is_rf = config.is_rf
        if config.is_rf:
            self.table = self._create_table()
        else:
            self.table = self._hashing_table_f()
        
        
    def _create_table(self, replacement = True):
        m = int(math.sqrt(self.input_dim)) * self.K // 2
        if m < 32: #split one node by evaulating at least 8 features
            m = 32
        if replacement:
            np.random.seed(0)
            return np.random.randint(0, self.input_dim, size = (self.L, m))
        else:
            import random
            random.seed(0)
            return np.array([random.sample(range(0, self.input_dim), m) for l in range(self.L)],dtype = np.int32)
        
    def _hashing_table_f(self, replacement = True):
        m = int(math.sqrt(self.input_dim))
        if m < 16: #if the depth of binary tree is 3, the mimimun features is used to get one tree is 15.
            m = 16
        if replacement:
            np.random.seed(3456)
            return np.random.randint(0, self.input_dim, size = (self.L, self.K, m))
        else:
            import random
            random.seed(3456)
            return np.array([[random.sample(range(0, self.input_dim), m) for k in range(self.K)] \
                             for l in range(self.L)],dtype = np.int32)
        
    def learning_rpst(self, trainX, trainy, save_dir='./'):
        if self.is_rf:
            self.learning_hashing_by_rf = RF_hashing(trainX, trainy, self.table, self.K, self.L, self.max_depth)
        else:
            self.learning_hashing_by_rf = F_hashing(trainX, trainy, self.table, self.K, self.L, self.max_depth)
        save_path = os.path.join(save_dir, 'data.represent')
        self._learning_map(save_path)
        
    def _learning_map(self, save_path):
        if os.path.exists(save_path):
            self.learning_hashing_by_rf.load_params(save_path)
        else:
            self.learning_hashing_by_rf.learning_map()
            self.learning_hashing_by_rf.dump_params(save_path)
            
class nested_svm(object):
    def __init__(self, config):
        self.L = config.L
        self._clf_cluster = [CalibratedClassifierCV(svm.LinearSVC()) for _l in range(self.L)]
        self.clf = CalibratedClassifierCV(svm.LinearSVC())#predict the probobility of malicious class
        
    def learn(self, input_rpst, datay):
        assert(len(input_rpst.shape) == 3)
        for _l in range(self.L):
            self._clf_cluster[_l].fit(input_rpst[_l], datay)
        print("the second step.")
        logists_train = np.array([self._clf_cluster[_l].predict_proba(input_rpst[_l])\
                         for _l in range(self.L)]
                                )
        logists_train = np.squeeze(logists_train[:,:,1]).transpose(1, 0)
        self.clf.fit(logists_train, datay)
    def predict(self, test_rpst, testy):
        logists = [self._clf_cluster[_l].predict_proba(test_rpst[_l]) \
                   for _l in range(self.L)]
        logists = np.squeeze(np.array(logists)[:,:,1]).transpose(1, 0)
        #self.clf.predict(logists)
        return self.clf.score(logists, testy), self.clf.predict_proba(logists)[:,1]
    
    def dump(self, save_dir):
        save_dir = os.path.join(save_dir, 'lh_svm/')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'nested_svm.pkl')
        joblib.dump([self._clf_cluster, self.clf], save_path)
        
    def load(self, load_dir):
        load_dir = os.path.join(load_dir, 'lh_svm/')
        load_path = os.path.join(load_dir, 'nested_svm.pkl')
        self._clf_cluster, self.clf = joblib.load(load_path)
        
class nested_lr(object):
    def __init__(self, config):
        self.L = config.L
        self._clf_cluster = [LogisticRegression() for _l in range(self.L)]
        self.clf = LogisticRegression()
        
    def learn(self, input_rpst, datay):
        assert(len(input_rpst.shape) == 3)
        for _l in range(self.L):
            self._clf_cluster[_l].fit(input_rpst[_l], datay)
        print("the second step.")
        logists_train = np.array([self._clf_cluster[_l].predict_proba(input_rpst[_l])\
                         for _l in range(self.L)]
                                )
        logists_train = np.squeeze(logists_train[:,:,1]).transpose(1, 0)
        self.clf.fit(logists_train, datay)
    def predict(self, test_rpst, testy):
        logists = [self._clf_cluster[_l].predict_proba(test_rpst[_l]) \
                   for _l in range(self.L)]
        logists = np.squeeze(np.array(logists)[:,:,1]).transpose(1, 0)
        #self.clf.predict(logists)
        return self.clf.score(logists, testy)
    
    def dump(self, save_dir):
        save_dir = os.path.join(save_dir, 'lh_lr/')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'nested_lr.pkl')
        joblib.dump([self._clf_cluster, self.clf], save_path)
        
    def load(self, load_dir):
        load_dir = os.path.join(load_dir, 'lh_lr/')
        load_path = os.path.join(load_dir, 'nested_lr.pkl')
        self._clf_cluster, self.clf = joblib.load(load_path)
        
class nested_dt(object):
    def __init__(self, config):
        self.L = config.L
        self._clf_cluster = [DecisionTreeClassifier(max_depth = 5, min_samples_leaf=1, random_state= 0) for _l in range(self.L)]
        self.clf = DecisionTreeClassifier()
        
    def learn(self, input_rpst, datay):
        assert(len(input_rpst.shape) == 3)
        for _l in range(self.L):
            self._clf_cluster[_l].fit(input_rpst[_l], datay)
        print("the second step.")
        logists_train = np.array([self._clf_cluster[_l].predict_proba(input_rpst[_l])\
                         for _l in range(self.L)]
                                )
        logists_train = np.squeeze(logists_train[:,:,1]).transpose(1, 0)
        self.clf.fit(logists_train, datay)
    def predict(self, test_rpst, testy):
        logists = [self._clf_cluster[_l].predict_proba(test_rpst[_l]) \
                   for _l in range(self.L)]
        logists = np.squeeze(np.array(logists)[:,:,1]).transpose(1, 0)
        #self.clf.predict(logists)
        return self.clf.score(logists, testy)
    
    def dump(self, save_dir):
        save_dir = os.path.join(save_dir, 'lh_dt/')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'nested_dt.pkl')
        joblib.dump([self._clf_cluster, self.clf], save_path)
        
    def load(self, load_dir):
        load_dir = os.path.join(load_dir, 'lh_dt/')
        load_path = os.path.join(load_dir, 'nested_dt.pkl')
        self._clf_cluster, self.clf = joblib.load(load_path)            
        
def trainSVM(config, train_rpst, trainy):
    robust_svm = nested_svm(config)
    robust_svm.learn(train_rpst, trainy)
    robust_svm.dump(save_dir)

def testSVM(config, test_rpst, testy):
    robust_svm = nested_svm(config)
    robust_svm.load(save_dir)
    acc, probs = robust_svm.predict(test_rpst, testy)
    return acc,probs

def trainlr(config, train_rpst, trainy):
    robust_lr = nested_lr(config)
    robust_lr.learn(train_rpst, trainy)
    robust_lr.dump(save_dir)

def testlr(config, test_rpst, testy):
    robust_lr = nested_lr(config)
    robust_lr.load(save_dir)
    acc = robust_lr.predict(test_rpst, testy)
    return acc

def traindt(config, train_rpst, trainy):
    robust_dt = nested_dt(config)
    robust_dt.learn(train_rpst, trainy)
    robust_dt.dump(save_dir)

def testdt(config, test_rpst, testy):
    robust_dt = nested_lr(config)
    robust_dt.load(save_dir)
    acc = robust_dt.predict(test_rpst, testy)
    return acc
    
        
if __name__ == '__main__':
    trainX = utils.readdata_np(TRAIN_DATA_PATH)
    trainy = utils.readdata_np(TRAIN_LABEL_PATH)[:,0]
    testX = utils.readdata_np(TEST_DATA_PATH)
    testy = utils.readdata_np(TEST_LABEL_PATH)[:,0]
    testX_adv = utils.readdata_np(ADV_DATA_PATH)
    testy_adv = utils.readdata_np(ADV_LABEL_PATH)[:,0]
    mal_data = utils.readdata_np(MAL_DATA_PATH)
    ben_data = utils.readdata_np(BEN_DATA_PATH)
    input_dim = trainX.shape[1]
    _config = {
            'save_dir': save_dir,
            'input_dim' : input_dim,
            'K' : K,  
            'L' : L,
	    'max_depth': 2,
            'is_rf': False
        }
    config = utils.ParamWrapper(_config)
    lr_rf = lr_rpst(config)
    lr_rf.learning_rpst(trainX, trainy, save_dir = save_dir)
    train_rpst = lr_rf.learning_hashing_by_rf.hashing_func(trainX).transpose(1, 0, 2)
    
    trainSVM(config, train_rpst, trainy)
    #test
    #test_rpst = lr_rf.learning_hashing_by_rf.hashing_func(testX).transpose(1, 0, 2)
    adv_rpst = lr_rf.learning_hashing_by_rf.hashing_func(testX_adv).transpose(1, 0, 2)
    mal_rpst = lr_rf.learning_hashing_by_rf.hashing_func(mal_data).transpose(1, 0, 2)
    ben_rpst = lr_rf.learning_hashing_by_rf.hashing_func(ben_data).transpose(1, 0, 2)
    maly = np.zeros((mal_data.shape[0],))
    maly[:] = 1.
    beny = np.zeros((ben_data.shape[0],))
    beny[:] = 0.
    fnr = 1. - testSVM(config, mal_rpst, maly)
    fpr = 1. - testSVM(config, ben_rpst, beny)
    acc = float((1. - fnr) * mal_data.shape[0] + (1. - fpr) * ben_data.shape[0]) / (mal_data.shape[0] + ben_data.shape[0])
    print("Accuracy of svm is %.5f, with false negative rate %.5f and false positive rate %.5f" % (acc, fnr, fpr))
    print("Accuracy of svm is %.5f" % testSVM(config, adv_rpst, testy_adv))
    
    trainlr(config, train_rpst, trainy)
    fnr = 1. - testlr(config, mal_rpst, maly)
    fpr = 1. - testlr(config, ben_rpst, beny)
    acc = float((1. - fnr) * mal_data.shape[0] + (1. - fpr) * ben_data.shape[0]) / (mal_data.shape[0] + ben_data.shape[0])
    #print("Accuracy of logit reg is %.5f" % testlr(config, test_rpst, testy))
    print("Accuracy of l_r is %.5f, with false negative rate %.5f and false positive rate %.5f" % (acc, fnr, fpr))
    print("Accuracy of logit reg is %.5f" % testlr(config, adv_rpst, testy_adv))
    
    traindt(config, train_rpst, trainy)
    fnr = 1. - testdt(config, mal_rpst, maly)
    fpr = 1. - testdt(config, ben_rpst, beny)
    acc = float((1. - fnr) * mal_data.shape[0] + (1. - fpr) * ben_data.shape[0]) / (mal_data.shape[0] + ben_data.shape[0])
    #print("Accuracy of logit reg is %.5f" % testlr(config, test_rpst, testy))
    print("Accuracy of l_r is %.5f, with false negative rate %.5f and false positive rate %.5f" % (acc, fnr, fpr))
    print("Accuracy of logit reg is %.5f" % testdt(config, adv_rpst, testy_adv))
