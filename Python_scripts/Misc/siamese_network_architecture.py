# -*- coding: utf-8 -*-
"""
@author: viveksagar (VivekSagar2016@u.northwestern.edu)
Inspired by Youngwook Paul Kwon's code for analysing MNIST data using supervised Siamese.
This codes trains the Siamese Network in an unsupervised way with a different loss function..
"""

import tensorflow as tf 
class siamese:
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, shape=(1171,21))
        self.x2 = tf.placeholder(tf.float32, shape=(1171,21))
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)
    def network(self, x):
        fc1 = self.fc_layer(x, 128, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 64, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 1, "fc3")
        return fc3
    def fc_layer(self, bottom, n_weight, name):
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight]))
        fc = tf.add(tf.matmul(bottom, W), b)
        return fc        
    def loss(self):
        diff = tf.sub(self.o1, self.o2)
        diff_mean = tf.reduce_mean(diff, 1) 
        diff_var = tf.sub(tf.reduce_mean(tf.pow(diff,2),1),tf.pow(diff_mean,2))
        coeff_var = tf.div(tf.sqrt(diff_var),diff_mean)
        loss= tf.reduce_sum(coeff_var)
        return loss
        
#    with tf.variable_scope("image_filters") as scope:
#    result1 = my_image_filter(image1)
#    scope.reuse_variables()
#    result2 = my_image_filter(image2)
