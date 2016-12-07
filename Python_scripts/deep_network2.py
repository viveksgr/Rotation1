# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:49:33 2016

@author: viveksagar
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import cross_validation

with np.load('Pre_processed.npz') as data:
    X1 = data['X1']
    X2 = data['X2']
    duration = data['X3']
   
all_X1=X1
all_X2=X2   
all_duration = duration

num_col = X1.shape[1]
X3 = np.concatenate((X1,X2,(np.matrix(duration)).T), axis=1)
np.random.shuffle(X3)
X1 = X3[:,0:num_col]
X2 = X3[:,num_col:-1]
duration = np.squeeze(np.asarray(X3[:,-1])) 

batch_size = 100
length = len(X1)
split_percent = 80
split = (split_percent/100)*length
split = (split-np.mod(split,batch_size)).astype(np.int64)       
split2 = (length-np.mod(length,batch_size)).astype(np.int64)
        
X1_train = X1[0:split,:]
X2_train = X2[0:split,:]
X1_test = X1[split:split2,:]
X2_test = X2[split:split2,:]
duration_train = duration[0:split]
duration=duration[split:split2]
duration = duration.astype(np.float64)
    
   
sz = np.shape(X1)

test_size = split2-split
hm_epochs = 1000
nn1 = 64
nn2 = 32
nn3 = 8
nn4 = 16
nn5 = 1



# Network structure and the loss function.
class siamese:
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, shape=(batch_size,sz[1]))
        self.x2 = tf.placeholder(tf.float32, shape=(batch_size,sz[1]))
        self.x3 = tf.placeholder(tf.float32, shape=(batch_size))
        self.x1_test = tf.placeholder(tf.float32, shape=(test_size,sz[1]))
        self.x2_test = tf.placeholder(tf.float32, shape=(test_size,sz[1]))
        self.dropout_f = tf.placeholder("float")        
        
        with tf.variable_scope("siamese") as scope:
            self.o1= self.build_model_mlp(self.x1, self.dropout_f)
            scope.reuse_variables()
            self.o2 = self.build_model_mlp(self.x2, self.dropout_f)
            
        with tf.variable_scope("siamese", reuse = True) as scope:
            self.o1_test= self.build_model_mlp(self.x1_test, self.dropout_f)
            scope.reuse_variables()
            self.o2_test = self.build_model_mlp(self.x2_test, self.dropout_f)
      
        #Loss
        dur = self.x3
        diff = tf.div(tf.sub(self.o1, self.o2),dur)
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
        l3 = tf.nn.dropout(l3,_dropout)
        l4 = self.mlp(l3,nn3,nn4,"l4")
        l4 = tf.nn.dropout(l4,_dropout)
        l5 = self.mlp(l4,nn4,nn5,"l5")
#        l5 = tf.nn.dropout(l5,_dropout)
#        l6 = self.mlp(l5,nn5,nn6,"l6")
        return l5        

    def mlp(self, input_,input_dim,output_dim,name):
        with tf.variable_scope(name, reuse=None):
            w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
            b = tf.get_variable('b',[1,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.00,stddev=0.02))
#            return tf.nn.softsign(tf.add(tf.matmul(input_,w),b))
            return tf.add(tf.matmul(input_,w),b)
#                                           
# Training and executing.
def train_neural_network(xx1,xx2,xx1_test,xx2_test,dur_tr):
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
            while i < len(xx1):
                start = i
                end = i+batch_size
                i+=batch_size
                batch_x1 = xx1[start:end,:]
                batch_x2 = xx2[start:end,:]
                batch_x3 = dur_tr[start:end]
                _, c, p1, p2 = sess.run([optimizer, model.loss, model.o1, model.o2], feed_dict={model.x1: batch_x1,model.x2: batch_x2,model.x3: batch_x3, model.dropout_f: 0.95})
                epoch_loss += c
                O1 = np.append(O1, p1)
                O2 = np.append(O2, p2)
            print('Epoch', epoch+1, '/',hm_epochs,'Loss:',epoch_loss, c)
            
           
        O1_test = model.o1_test
        O2_test = model.o2_test
        O1_test_= O1_test.eval({model.x1_test:xx1_test, model.dropout_f:0.9})
        O2_test_= O2_test.eval({model.x2_test:xx2_test, model.dropout_f:0.9})      
    return O1, O2, O1_test_, O2_test_
    return O1, O2
    
    
    
    
                
[O1, O2, O1_t, O2_t] = train_neural_network(X1_train, X2_train, X1_test, X2_test, duration_train)
O1 = np.delete(O1, (0))
O2 = np.delete(O2, (0))
O3=O2-O1
O3_t = np.squeeze(O2_t-O1_t)
O3_t=O3_t/np.std(O3_t)
np.savez("Deep_data_dur", O1=O1_t, O2=O2_t, O3=O3_t)
np.savez("Fim_data1",X1=X1_test,X2=X2_test,duration=duration)


t = len(np.where(O3_t<0)[0])
fim = np.sum(X2_test-X1_test,axis=1)
t_2 = len(np.where(fim<0)[0])


n, bins, patches = plt.hist(O3_t, 40, normed=0, facecolor='green', alpha=0.85)
plt.xlabel('Score')
plt.ylabel('#')
plt.title('Softsign Activation')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('deep_score.png', bbox_inches='tight')

plt.figure()
plt.scatter(fim,O3_t)
plt.xlabel('Fim')
plt.ylabel('Deep')
fig2=plt.gcf()
fig2.set_size_inches(18.5,10.5)
fig2.savefig('fim_deep.png')

plt.figure()
n, bins, patches = plt.hist(fim, 40, normed=0, facecolor='green', alpha=0.85)
plt.xlabel('Score')
plt.ylabel('#')
plt.title('Fim score')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('fim.png', bbox_inches='tight')


def ghadha_ghoda(vec, week_l):
    week_l = week_l.astype(np.int64)
    ind = np.argsort(week_l)
    sorted_week = week_l[ind]
    sorted_diff = sorted_week-np.roll(sorted_week,1)
    start_ind = np.where(sorted_diff!=0)[0]
    sorted_diff2 = np.roll(sorted_week,-1)-sorted_week
    end_ind = np.where(sorted_diff2!=0)[0]
    x_axis = sorted_week[end_ind]
    sorted_data = vec[ind]
    mean_list = []
    stdv_list = []
    for ii in range(len(x_axis)):     
        if start_ind[ii]<end_ind[ii]:
            temp_dev = np.std(sorted_data[start_ind[ii]:end_ind[ii]])
            temp_mean=np.mean(sorted_data[start_ind[ii]:end_ind[ii]])
        else:
            temp_mean = sorted_data[start_ind[ii]]
            temp_dev=0
        mean_list.append(temp_mean)
        stdv_list.append(temp_dev)
    mean_list = np.asarray(mean_list)
    length = end_ind-start_ind
    stdv_list = np.asarray(stdv_list)/np.sqrt(length)
    return mean_list, stdv_list, x_axis, length
#
#duration = np.floor_divide(duration,7)
#all_duration = np.floor_divide(all_duration,7)

[mean_vec, stdv_vec, x_axis, length]=ghadha_ghoda(O3_t,duration)
[mean_vec2, stdv_vec2, x_axis2, length2]=ghadha_ghoda(fim,duration)

plt.figure()
plt.errorbar(x_axis, mean_vec, yerr=stdv_vec)
plt.plot(x_axis,np.log(length),'r')
plt.title("Deep-duration-score vs Duration")
plt.xlabel('Duration')
plt.ylabel('Deep_score')
fig2=plt.gcf()
fig2.set_size_inches(18.5,10.5)
fig2.savefig('deep_vs_dur.png')

plt.figure()
plt.errorbar(x_axis2, mean_vec2, yerr=stdv_vec2)
plt.plot(x_axis2,np.log(length2),'r')
plt.title("DeltaFim-score vs Duration")
plt.xlabel('Duration')
plt.ylabel('Fim_score')
fig2=plt.gcf()
fig2.set_size_inches(18.5,10.5)
fig2.savefig('Fim_vs_dur.png')

fim2 = np.squeeze(np.asarray(np.sum(all_X2[:,:21]-all_X1[:,:21],axis=1)))
[mean_vec3, stdv_vec3, x_axis3, length3]=ghadha_ghoda(fim2,all_duration)

plt.figure()
plt.errorbar(x_axis3, mean_vec3, yerr=stdv_vec3)
#plt.plot(x_axis3,np.max(mean_vec3)*length3/np.max(length3),'r')
plt.plot(x_axis3,np.log(length3),'r')
plt.title("DeltaFim-score (for all patients) vs Duration")
plt.xlabel('Duration')
plt.ylabel('Fim_score')
fig2=plt.gcf()
fig2.set_size_inches(18.5,10.5)
fig2.savefig('all_Fim_vs_dur.png')
