"""
Created on Tue Nov  1 12:58:12 2016
@author: viveksagar (VivekSagar2016@u.northwestern.edu)
Inspired by Youngwook Paul Kwon's code for analysing MNIST data using supervised Siamese.
This codes trains the Siamese Network in an unsupervised way with a different loss function.
"""
import tensorflow as tf
import numpy as np
import pandas as pd

# Get the preprocessed data. X1 and X2 are the entry and exit data.
X1 = pd.ExcelFile("entry_data.xlsx")
entry_data = X1.parse("Sheet1")
X2 = pd.ExcelFile("exit_data.xlsx")
exit_data = X2.parse("Sheet1")
X1 = np.array(entry_data.astype(np.float32))
X2 = np.array(exit_data.astype(np.float32))
assert np.shape(X1) == np.shape(X2)
sz = np.shape(X1)


# Network structure and the loss function.
#class siamese:
#    def __init__(self):
#        self.x1 = tf.placeholder(tf.float32, shape=sz)
#        self.x2 = tf.placeholder(tf.float32, shape=sz)
#        with tf.variable_scope("siamese") as scope:
#            self.o1 = self.network(self.x1)
#            scope.reuse_variables()
#            self.o2 = self.network(self.x2)
#    def network(self, x):
#        fc1 = self.fc_layer(x, 128, "fc1")
#        ac1 = tf.nn.relu(fc1)
#        fc2 = self.fc_layer(ac1, 64, "fc2")
#        ac2 = tf.nn.relu(fc2)
#        fc3 = self.fc_layer(ac2, 2, "fc3")
#        return fc3
#    def fc_layer(self, bottom, n_weight, name):
#        assert len(bottom.get_shape()) == 2
#        n_prev_weight = bottom.get_shape()[1]
#        initer = tf.truncated_normal_initializer(stddev=0.01)
#        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
#        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight]))
#        fc = tf.add(tf.matmul(bottom, W), b)
#        return fc        
#    def loss(self):
#        diff = tf.sub(self.o1, self.o2)
#        diff_mean = tf.reduce_mean(diff, 1) 
#        diff_var = tf.sub(tf.reduce_mean(tf.pow(diff,2),1),tf.pow(diff_mean,2))
#        coeff_var = tf.div(tf.sqrt(diff_var),diff_mean)
#        loss= tf.reduce_sum(coeff_var)
#        return loss

import siamese_network
siamese = siamese_network.siamese();

        
# Training and executing.
batch_size = 100
hm_epochs = 10
def train_neural_network(xx1,xx2):
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(siamese.loss)
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(xx1)-batch_size:
				start = i
				end = i+batch_size
				batch_x1 = xx1[start:end]
				batch_x2 = xx2[start:end]
    
				_, c = sess.run([optimizer, siamese.loss], feed_dict={siamese.x1: batch_x1,
				                                              siamese.x2: batch_x2})
                                                  
                                                  
				epoch_loss += c
				i+=batch_size				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
                
train_neural_network(X1,X2)