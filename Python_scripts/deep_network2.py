"""
@author: viveksagar (VivekSagar2016@u.northwestern.edu)
"""
import tensorflow as tf
import numpy as np
import pandas as pd

# Get the preprocessed data. X1 and X2 are the entry and exit data.
X1 = pd.ExcelFile("entry_data.xlsx")
entry_data = X1.parse("Sheet1")
X2 = pd.ExcelFile("exit_data.xlsx")
exit_data = X2.parse("Sheet1")
X1 = np.array(entry_data.astype(np.float64))
X2 = np.array(exit_data.astype(np.float64))
assert np.shape(X1) == np.shape(X2)
sz = np.shape(X1)


# Network structure and the loss function.
class siamese:
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, shape=(100,21))
        self.x2 = tf.placeholder(tf.float32, shape=(100,21))
        self.dropout_f = tf.placeholder("float")
        
        
        with tf.variable_scope("siamese") as scope:
            self.o1= self.build_model_mlp(self.x1, self.dropout_f)
            scope.reuse_variables()
            self.o2 = self.build_model_mlp(self.x2, self.dropout_f)                   
      
        #Loss
        diff = tf.sub(self.o1, self.o2)
        diff_mean = tf.reduce_mean(diff, 1) 
        diff_var = tf.sub(tf.reduce_mean(tf.pow(diff,2),1),tf.pow(diff_mean,2))
        coeff_var = tf.div(tf.sqrt(diff_var),diff_mean)
        self.loss= tf.reduce_sum(coeff_var)      

    def build_model_mlp(self, X_,_dropout):
        model = self.mlpnet(X_,_dropout)
        return model

    def mlpnet(self, x,_dropout):
        l1 = self.mlp(x,21,128,"l1")
        l1 = tf.nn.dropout(l1,_dropout)
        l2 = self.mlp(l1,128,64,"l2")
        l2 = tf.nn.dropout(l2,_dropout)
        l3 = self.mlp(l2,64,1,"l3")
        return l3        

    def mlp(self, input_,input_dim,output_dim,name):
        with tf.variable_scope(name, reuse=None):
            w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
            return tf.nn.relu(tf.matmul(input_,w))
                                                      
# Training and executing.
batch_size = 100
hm_epochs = 10
def train_neural_network(xx1,xx2):
    model = siamese()
    print("Model ok")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss=model.loss)
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
                _, c = sess.run([optimizer, model.loss], feed_dict={model.x1: batch_x1,model.x2: batch_x2, model.dropout_f: 0.9})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
                
train_neural_network(X1,X2)