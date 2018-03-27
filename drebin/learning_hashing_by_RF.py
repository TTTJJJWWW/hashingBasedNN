#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:37:29 2018
learning hashing codes with random forest
@email: dli@fiu.edu
@author: dli
"""
import numpy as np
import matplotlib.pylab as plt

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import math

#Forest 
#Note: parallel training should be supported
from sklearn.tree import DecisionTreeClassifier 
class Forest(object):
    def __init__(self, k, max_depth = 3, criterion = 'entropy'):
        self.k = k #number of trees
        self.max_depth = max_depth
        self.criterion = criterion
        self.dts = [DecisionTreeClassifier(\
                                           criterion = self.criterion,
                                           max_depth = self.max_depth
                                          )
                    for _k in range(self.k)
                   ]
    def fit(self, random_features, labels):
        assert len(random_features.shape)==3 and random_features.shape[0] == self.k
        self.features = random_features #[k, n_samples, m]
        self.labels = labels
        for _k in range(self.k):
            _X = self.features[_k]
            _Y = labels
            self.dts[_k].fit(_X, _Y)
        
    def applys(self, inputs):
        assert len(inputs.shape) == 3 #[k, sampels, m]
        idxs = []
        for _k in range(self.k):
            _X = inputs[_k]
            idx = self.dts[_k].apply(_X)
            idxs.append(idx)
        return np.vstack(idxs).transpose(1, 0)
    
    def dump(self):
        
        return
    def load(self):
        return
    
#forest hashing functions
from sklearn.preprocessing import OneHotEncoder
class F_hashing(object):
    def __init__(self, inputs, labels, table, k, l, max_depth = 3):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels
        self.table = table
        self.k = k
        self.l = l
        self.max_depth = max_depth
        self.rfs = [Forest(\
                           k = self.k,
                           max_depth = self.max_depth,
                           criterion = 'entropy'
                          ) \
                    for _l in range(l)]
        n_values = [math.pow(2, self.max_depth + 1) - 1 for _k in range(self.k)]
        self.rfs_one = [OneHotEncoder(n_values= n_values) for _l in range(l)]
        
    def learning_map(self):
        rf_hashing_logits = []
        for _l in range(self.l):
            #features
            random_features = self.inputs[:,self.table[_l]].transpose(1, 0, 2)
            self.rfs[_l].fit(random_features, self.labels)
            self.rfs_one[_l].fit(self.rfs[_l].applys(random_features))
            print self.rfs[_l].applys(random_features)[0, :10]
    
    def hashing_func(self, inputs_):
        #assert inputs_.shape[1] == self.inputs.shape[1]
        hashing_encode = \
        [self.rfs_one[_l].transform(
            self.rfs[_l].applys(inputs_[:, self.table[_l]].transpose(1, 0, 2))).toarray()\
                                             for _l in range(self.l)]
        hashing_encode = np.transpose(np.array(hashing_encode), (1, 0, 2))
        
        return hashing_encode #(n_samples, l, l * (2^d -2 or oneEncoding_length) )
    
    def dump_params(self, save_path):
        save_dir = os.path.dirname(save_path)
        file_name = os.path.basename(save_path)
        joblib.dump(self.rfs, save_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        joblib.dump(self.rfs_one, save_path_2)
        #save model 
        '''
        for _l in range(self.l):
            file_namel = file_name + str(_l)
            save_path = os.path.join(save_dir, file_namel)
            joblib.dump(self.rfs[_l], save_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        joblib.dump(self.rfs_one, save_path_2)
        '''
    
    def load_params(self, load_path):
        save_dir = os.path.dirname(load_path)
        file_name = os.path.basename(load_path)
        self.rfs = joblib.load(load_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        self.rfs_one = joblib.load(save_path_2)
        
        #save model 
        '''
        for _l in range(self.l):
            file_namel = file_name + str(_l)
            save_path = os.path.join(save_dir, file_namel)
            self.rfs[_l] = joblib.load(save_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        self.rfs_one = joblib.load(save_path_2)  
        '''

class RF_hashing(object):
    def __init__(self, inputs, labels, tables, k, l, max_depth = 3):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels
        self.tables = tables
        self.k = k
        self.l = l
        self.max_depth = max_depth
        self.rfs = [RandomForestClassifier(\
                            n_estimators = self.k,
                            max_depth = 3,
                            criterion = 'entropy',
                            verbose = True,
                            n_jobs = -1) \
                    for _l in range(l)]
        
        n_values = [math.pow(2, self.max_depth + 1) -1 for _k in range(self.k)]
        self.rfs_one = [OneHotEncoder(n_values = n_values) for _l in range(self.l)]
        
    def learning_map(self):
        rf_hashing_logits = []
        for _l in range(self.l):
            #features
            random_features = self.inputs[:,self.tables[_l]]
            self.rfs[_l].fit(random_features, self.labels)
            self.rfs_one[_l].fit(self.rfs[_l].apply(random_features))
    
    def hashing_func(self, inputs_):
        #assert inputs_.shape[1] == self.inputs.shape[1]
        hashing_encode = \
        [self.rfs_one[_l].transform(
            self.rfs[_l].apply(inputs_[:, self.tables[_l]])).toarray() for _l in range(self.l)]
        hashing_encode = np.transpose(np.array(hashing_encode), (1, 0, 2))
        
        return hashing_encode #(n_samples, l, n_trees * (2^d -2) )
    
    def dump_params(self, save_path):
        save_dir = os.path.dirname(save_path)
        file_name = os.path.basename(save_path)
        joblib.dump(self.rfs, save_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        joblib.dump(self.rfs_one, save_path_2)
        #save model 
        '''
        for _l in range(self.l):
            file_namel = file_name + str(_l)
            save_path = os.path.join(save_dir, file_namel)
            joblib.dump(self.rfs[_l], save_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        joblib.dump(self.rfs_one, save_path_2)
        '''
    def load_params(self, load_path):
        save_dir = os.path.dirname(load_path)
        file_name = os.path.basename(load_path)
        self.rfs = joblib.load(load_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        self.rfs_one = joblib.load(save_path_2)
        #save model 
        '''
        for _l in range(self.l):
            file_namel = file_name + str(_l)
            save_path = os.path.join(save_dir, file_namel)
            self.rfs[_l] = joblib.load(save_path)
        save_path_2 = os.path.join(save_dir, 'rf.onehot')
        self.rfs_one = joblib.load(save_path_2)
        '''


