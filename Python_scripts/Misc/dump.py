# -*- coding: utf-8 -*-
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D) 
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.sub(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.sub(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.mul(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.sub(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.sub(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.sub(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.mul(labels_t, eucd, name="y_x_eucd")
        neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
        


# prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese();
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
new = True
model_ckpt = 'model.ckpt'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# start training
if new:
    for step in range(100000):
        batch_x1, batch_y1 = mnist.train.next_batch(128)
        batch_x2, batch_y2 = mnist.train.next_batch(128)
        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_x1, 
                            siamese.x2: batch_x2, 
                            siamese.y_: batch_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 10 == 0:
            print ('step %d: loss %.3f' % (step, loss_v))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, 'model.ckpt')
            embed = siamese.o1.eval({siamese.x1: mnist.test.images})
            embed.tofile('embed.txt')
else:
    saver.restore(sess, 'model.ckpt')

# visualize result
x_test = mnist.test.images.reshape([-1, 28, 28])
visualize.visualize(embed, x_test)

        
        
