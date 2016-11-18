import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import cross_validation

with np.load('Pre_processed.npz') as data:
    X1 = data['X1']
    X2 = data['X2']
    duration = data['X3']
    
duration = duration.astype(np.float64)
    
#with np.load('Fim_data.npz') as data2:
#    Fim_in = data2['In']
#    Fim_out = data2['Out']
        
#X1_train, X1_test, X2_train, X2_test = cross_validation.train_test_split(X1, X2, test_size=0.2)
X1_train = X1[0:800,:]
X2_train = X2[0:800,:]
X1_test = X1[800:1000,:]
X2_test = X2[800:1000,:]
duration_train = duration[0:800]
duration=np.delete(duration, np.s_[:800])
    
sz = np.shape(X1)
batch_size = 100
test_size = 200
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
#        self.x3_test = tf.placeholder(tf.float32, shape=(test_size,sz[1]))
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
            return tf.nn.softsign(tf.add(tf.matmul(input_,w),b))
#            return tf.add(tf.matmul(input_,w),b)
                                           
# Training and executing.
def train_neural_network(xx1,xx2,xx1_test,xx2_test,dur_tr):
    model = siamese()
    print("Model executed successfully.")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=model.loss)
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
                batch_x1 = xx1[start:end]
                batch_x2 = xx2[start:end]
                batch_x3 = dur_tr[start:end]
                _, c, p1, p2 = sess.run([optimizer, model.loss, model.o1, model.o2], feed_dict={model.x1: batch_x1,model.x2: batch_x2,model.x3: batch_x3, model.dropout_f: 0.87})
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
O3_t = O3_t/duration
np.savez("Deep_data2", O1=O1, O2=O2, O3=O3)

#with np.load('Deep_data1.npz') as data:
#    O11 = data['O1']
#    O21 = data['O2']
#    O31=O21-O11
#O = (O3-O31)

n, bins, patches = plt.hist(O3_t/(np.std(O3_t)), 50, normed=0, facecolor='green', alpha=0.8)
plt.xlabel('Score/duration')
plt.ylabel('#')
plt.title('Softsign Activation')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('7.png', bbox_inches='tight')

plt.figure()
plt.plot(duration, O3_t,'.'
)
plt.xlabel('Duration')
plt.ylabel('Score/duration')
fig2=plt.gcf()
fig2.set_size_inches(18.5,10.5)
fig2.savefig('Scatter_score.png')
