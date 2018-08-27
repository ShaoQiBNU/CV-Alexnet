########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_sets",one_hot=True)

########## set net hyperparameters ##########
learning_rate=0.0001
epochs=20
batch_size=128
display_step=30

########## set net parameters ##########
#### img shape:28*28
n_input=784 

#### 0-9 digits
n_classes=10

#### dropout probability
dropout=0.5

########## placeholder ##########
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])

##################### build net model ##########################

########## define conv process ##########
def conv2d(name,x,W,b,strides=1, padding='SAME'):
	x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding=padding)
	x=tf.nn.bias_add(x,b)
	return tf.nn.relu(x,name=name)

########## define pool process ##########
def maxpool2d(name, x, k=3, s=2, padding='SAME'):
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,s,s,1],padding=padding,name=name)

########## define norm process ##########
def norm(name, l_input, lsize=5):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.0001, beta=0.75, name=name)

########## set net parameters ##########
def weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

weights={
	'wc1': weight_var('wc1',[11,11,1,96]),
	'wc2': weight_var('wc2',[5,5,96,256]),
	'wc3': weight_var('wc3',[3,3,256,384]),
	'wc4': weight_var('wc4',[3,3,384,384]),
	'wc5': weight_var('wc5',[3,3,384,256]),
	'wd1': weight_var('wd1',[4*4*256,4096]),
	'wd2': weight_var('wd2',[4096,4096]),
	'out_w': weight_var('out_w',[4096,10])
}
biases={
	'bc1': bias_var('bc1',[96]),
	'bc2': bias_var('bc2',[256]),
	'bc3': bias_var('bc3',[384]),
	'bc4': bias_var('bc4',[384]),
	'bc5': bias_var('bc5',[256]),
	'bd1': bias_var('bd1',[4096]),
	'bd2': bias_var('bd2',[4096]),
	'out_b': bias_var('out_b',[n_classes])
}

##################### build net model ##########################

########## define net structure ##########
def alexnet(x, weights, biases, dropout):
	#### reshape input picture ####
	x=tf.reshape(x, shape=[-1,28,28,1])

	#### 1 conv ####
	## conv ##
	conv1=conv2d('conv1', x, weights['wc1'], biases['bc1'], padding='SAME')
	## pool ##
	pool1=maxpool2d('pool1',conv1,k=3, s=2, padding='SAME')
	## norm ##
	norm1=norm('norm1', pool1, lsize=5)

	#### 2 conv ####
	## conv ##
	conv2=conv2d('conv2', norm1, weights['wc2'], biases['bc2'], padding='SAME')
	## pool ##
	pool2=maxpool2d('pool2',conv2,k=3, s=2, padding='SAME')
	## norm ##
	norm2=norm('norm2', pool2, lsize=5)

	#### 3 conv ####
	## conv ##
	conv3=conv2d('conv3', norm2, weights['wc3'], biases['bc3'], padding='SAME')

	#### 4 conv ####
	## conv ##
	conv4=conv2d('conv4', conv3, weights['wc4'], biases['bc4'], padding='SAME')

	#### 5 conv ####
	## conv ##
	conv5=conv2d('conv5', conv4, weights['wc5'], biases['bc5'], padding='SAME')
	## pool ##
	pool5=maxpool2d('pool5',conv5,k=3, s=2, padding='SAME')
	## norm ##
	norm5=norm('norm5', pool5, lsize=5)

	#### 1 fc ####
	fc1=tf.reshape(norm5,[-1,weights['wd1'].get_shape().as_list()[0]])
	fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
	fc1=tf.nn.relu(fc1)

	## dropout ##
	fc1=tf.nn.dropout(fc1, dropout)

	#### 2 fc ####
	#fc2=tf.reshape(fc1,[-1,weights['wd2'].get_shape().as_list()[0]])
	fc2=tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'])
	fc2=tf.nn.relu(fc2)

	## dropout ##
	fc2=tf.nn.dropout(fc2, dropout)

	#### output ####
	out=tf.add(tf.matmul(fc2,weights['out_w']),biases['out_b'])
	return out

########## define model, loss and optimizer ##########

#### model ####
pred=alexnet(x, weights, biases, dropout)

#### loss ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    #### epoch 世代循环 ####
    for epoch in range(epochs+1):

        #### iteration ####
        for _ in range(mnist.train.num_examples//batch_size):

            step += 1

            ##### get x,y #####
            batch_x, batch_y=mnist.train.next_batch(batch_size)

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_x, y: batch_y})
                print("Epoch "+ str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))


    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples//batch_size):
        batch_x,batch_y=mnist.test.next_batch(batch_size)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
