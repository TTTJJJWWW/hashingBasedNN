#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:19:10 2018
@email: dli@fiu.edu
@author: dli
"""
import tensorflow as tf
import utils
import os
ADV_DATA_PATH = os.path.join('./dataset', "adv_smps.data")
ADV_LABEL_PATH = os.path.join('./dataset', "adv_smps.label")

def main(_):
    testX_adv = utils.readdata_np(ADV_DATA_PATH)
    testy_adv = utils.readdata_np(ADV_LABEL_PATH)
    
    test_adv_input = utils.DataProducer(testX_adv, testy_adv, 128, 30, "test_adv")
    for j in range(10):
        for i, x, y in test_adv_input.next_batch():
            print i
        test_adv_input.reset_cursor()
    
if __name__ == '__main__':
    tf.app.run()
