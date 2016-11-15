# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:13:42 2016

@author: viveksagar
"""

import tensorflow as tf
import numpy as np

with np.load('Pre_processed.npz') as data:
    X1 = data['X1']
    X2 = data['X2']
    
#assert np.shape(X1) == np.shape(X2)
sz = np.shape(X1)
batch_size = 100
nn1 = 64
nn2 = 32
nn3 = 1


# Network structure and the loss function.
class siamese:
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, shape=(batch_size,sz[1]))
        self.x2 = tf.placeholder(tf.float32, shape=(batch_size,sz[1]))
        self.dropout_f = tf.placeholder("float")        
        
        with tf.variable_scope("siamese") as scope:
            self.o1= self.build_model_mlp(self.x1, self.dropout_f)
            scope.reuse_variables()
            self.o2 = self.build_model_mlp(self.x2, self.dropout_f)
      
        #Loss
        diff = tf.sub(self.o1, self.o2)
        diff_mean = tf.reduce_mean(diff, 0) 
        diff_var = tf.sub(tf.reduce_mean(tf.pow(-diff,2),0),tf.pow(-diff_mean,2))
        coeff_var = tf.div(diff_mean,tf.sqrt(diff_var))
        self.loss= tf.reduce_sum(coeff_var)      

    def build_model_mlp(self, X_,_dropout):
        model = self.mlpnet(X_,_dropout)
        return model

    def mlpnet(self, x,_dropout):
        l1 = self.mlp(x,sz[1],nn1,"l1")
        l1 = tf.nn.dropout(l1,_dropout)
        l2 = self.mlp(l1,nn1,nn2,"l2")
        l2 = tf.nn.dropout(l2,_dropout)
        l3 = self.mlp(l2,nn2,nn3,"l3")
        return l3        

    def mlp(self, input_,input_dim,output_dim,name):
        with tf.variable_scope(name, reuse=None):
            w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
            return tf.nn.relu(tf.matmul(input_,w))
                                           
# Training and executing.
hm_epochs = 10
def train_neural_network(xx1,xx2):
    model = siamese()
    print("Model executed successfully.")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss=model.loss)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())	    
        for epoch in range(hm_epochs):
            epoch_loss = 0
            O1 = np.array([0])
            O2 = np.array([0])
            i=0
            while i < len(xx1)-batch_size:
                start = i
                end = i+batch_size
                batch_x1 = xx1[start:end]
                batch_x2 = xx2[start:end]
                _, c, p1, p2 = sess.run([optimizer, model.loss, model.o1, model.o2], feed_dict={model.x1: batch_x1,model.x2: batch_x2, model.dropout_f: 0.9})
                epoch_loss += c
                O1 = np.append(O1, p1)
                O2 = np.append(O2, p2)
                i+=batch_size
            print('Epoch', epoch+1, '/',hm_epochs,'Loss:',epoch_loss)
    return O1, O2
                
[O1, O2] = train_neural_network(X1,X2)
O1 = np.delete(O1, (0))
O2 = np.delete(O2, (0))
#np.savez("Deep_data", O1=O1, O2=O2)

#with np.load('Deep_data.npz') as data:
#    O1 = data['O1']
#    O2 = data['O2']
#parr= np.append(O1, O2) 
#parr2 = parr.reshape(2,len(parr)/2)
#parr2 = parr2.T
#np.savetxt('Deep_data.csv', parr2, delimiter=',')   

