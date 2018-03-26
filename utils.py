#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:37:36 2018
@email: dli@fiu.edu
@author: dli
"""

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import os
import warnings
import sys

import tensorflow as tf
import numpy as np
flags = tf.flags
FLAGS = flags.FLAGS

class ParamWrapper(object):
    def __init__(self, params):
        if not isinstance(params, dict):
            params = vars(params)
        self.params = params
        
    def __getattr__(self, name):
        val = self.params.get(name)
        if val is None:
            MSG = "Setting params ({}) is deprecated"
            warnings.warn(MSG.format(name))
            val = FLAGS.__getattr__(name)
        return val

def weight_init(shape, name):
    return tf.Variable(tf.random_uniform(shape, -tf.sqrt(6./(shape[0] + shape[-1])), tf.sqrt(6./(shape[0] + shape[-1]))), name= name)

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def byte_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def float_feature(value):
    return tf.train.Feature(floatlist = tf.train.FloatList(value = [value]))

def initialize_uninitilized_global_variables(sess):
    #from https://github.com/tensorflow/cleverhans/tree/master/cleverhans
    #List all global variables
    global_vars = tf.global_variables()
    #Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)
    
    #List all variables that were not initialialized previously
    not_initialized_vars = [var for (var, init) in 
                            zip(global_vars, is_initialized) if not init]
    
    #Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        
#============================IO==========================================
def readdata_np(data_path):
    with open(data_path, 'rb') as f_r:
        data = np.load(f_r)
    return data

def dumpdata_np(data_path, data):
    if not os.path.exists(data_path):
        os.makedirs(os.path.split(data_path)[0])
    with open(data_path, 'wb') as f_s:
        np.save(f_s, data)

class DataProducer(object):
    def __init__(self, dataX, datay, batch_size, n_epoches, name = 'train'):
        self.dataX = dataX
        self.datay = datay
        self.batch_size = batch_size
        self.mini_batches = self.dataX.shape[0] // self.batch_size +1
        self.name = name                                  
        self.cursor = 0
        
    def next_batch(self):
        while self.cursor < self.mini_batches:
            start_i = self.cursor * self.batch_size
            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            if start_i == end_i:
                break
            self.cursor = self.cursor + 1
            yield self.cursor, self.dataX[start_i:end_i], self.datay[start_i : end_i]
        
    def next_batch_dual(self):
        while self.cursor < self.mini_batches:
            start_i = self.cursor * self.batch_size
            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            if start_i == end_i:
                break
            self.cursor = self.cursor + 1
            yield self.cursor, start_i, end_i, self.dataX[start_i:end_i], self.datay[start_i : end_i]
              
        
    def reset_cursor(self):
        self.cursor = 0
        
    def get_current_cursor(self):
        return self.cursor  
