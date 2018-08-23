alexnet.py为alexnet的模型重现，输入数据为227-227-3。 
================================================

alexnet_MNIST.py 和 alexnet_MNIST2.py为利用Alexnet实现MNIST，第一层卷积和最后一层全连接层的参数做了调整，为了适应MNIST。
==============================================================================================================
# 一. 利用Alexnet的5层卷积结构实现MNIST识别（注意：层数传递时影像的大小很重要，直接决定网络是否正确）


参考李嘉璇的tensorflow和网上的代码，总结如下：
------------------------------------------

# 二.李嘉璇————定义5层网络结构，每层都是卷积+池化+规范化

## 1.卷积————步长为1， padding为 SAME

		########## define conv process ##########
		
		def conv2d(name,x,W,b,strides=1):
		
			x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
			
			x=tf.nn.bias_add(x,b)
			
			return tf.nn.relu(x,name=name)

## 2.pool————步长为2，padding为 SAME
		########## define pool process ##########
		
		def maxpool2d(name,x,k=2):
		
			return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)

## 3.规范化
		########## define norm process ##########
		
		def norm(name, l_input, lsize=4):
		
			return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)

## 4.网络参数—————卷积层的weights和biases的shape设置，即卷积核和偏移量

### (1).卷积核——————[filter_height, filter_width, in_channels/input_feature maps, channel_multiplier/output_feature maps] 

		########## set net parameters ##########
		
		weights={
			########## wc1 卷积核11*11*1*96， 输入为28*28*1，所以in_channels=1,96代表卷积核个数，表示有96个11*11*1的卷积核 #########
			'wc1': tf.Variable(tf.random_normal([11,11,1,96])),
			
			########## wc2 卷积核5*5*96*256， 输入为14*14*96，所以in_channels=96,256代表卷积核个数，表示有256个14*14*96的卷积核 #########
			'wc2': tf.Variable(tf.random_normal([5,5,96,256])),
			
			########## wc3 卷积核3*3*256*384， 输入为7*7*256，所以in_channels=256,384代表卷积核个数，表示有384个7*7*256的卷积核 #########
			'wc3': tf.Variable(tf.random_normal([3,3,256,384])),
			
			########## wc4 卷积核3*3*384*384， 输入为4*4*384，所以in_channels=384,384代表卷积核个数，表示有384个3*3*384的卷积核 #########
			'wc4': tf.Variable(tf.random_normal([3,3,384,384])),
			
			########## wc5 卷积核3*3*384*256， 输入为4*4*384，所以in_channels=384,256代表卷积核个数，表示有256个3*3*384的卷积核 #########
			'wc5': tf.Variable(tf.random_normal([3,3,384,256])),
			
			########## wd1 2*2*256*4096， 输入为2*2*256，所以将其展平则为1*(2*2*256),4096表示全连接层神经元的个数 #########
			'wd1': tf.Variable(tf.random_normal([2*2*256,4096])),
			
			########## wd2 4096*4096， 输入为[2*2*256,4096]，4096表示有4096个神经元 #########
			'wd2': tf.Variable(tf.random_normal([4096,4096])),
			
			########## out 4096,10， 输入为[4096,4096]，10表示有10类————> 0-9 #########
			'out': tf.Variable(tf.random_normal([4096,10]))
			}
		
## (2).偏移——————[channel_multiplier/output_feature maps] 		
		biases={
			'bc1': tf.Variable(tf.random_normal([96])),
			
			'bc2': tf.Variable(tf.random_normal([256])),
			
			'bc3': tf.Variable(tf.random_normal([384])),
			
			'bc4': tf.Variable(tf.random_normal([384])),
			
			'bc5': tf.Variable(tf.random_normal([256])),
			
			'bd1': tf.Variable(tf.random_normal([4096])),
			
			'bd2': tf.Variable(tf.random_normal([4096])),
			
			'out': tf.Variable(tf.random_normal([n_classes]))
			}
## 5.网络结构
		def alex_net(x, weights, biases, dropout):
			
			#### reshape input picture 输入数字是1*784的数据，将其reshape成28*28的影像 ####
			x=tf.reshape(x, shape=[-1,28,28,1])


			#### 1 conv 第1层卷积 ####					
			## conv size变化为 28*28———> 28*28，ceil(28/1) ##
			conv1=conv2d('conv1', x, weights['wc1'], biases['bc1'])
			
			## pool size变化为 28*28———> 14*14，ceil(14/2) ##
			pool1=maxpool2d('pool1',conv1,k=2)
			
			## norm size变化为 14*14———> 14*14 ##
			norm1=norm('norm1', pool1, lsize=4)


			#### 2 conv 第2层卷积####			
			## conv size变化为 14*14———> 14*14 ，ceil(14/1) ##
			conv2=conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
			
			## pool size变化为 14*14———> 7*7 ，ceil(14/2) ##
			pool2=maxpool2d('pool2',conv2,k=2)
			
			## norm size变化为 7*7———> 7*7  ##
			norm2=norm('norm2', pool2, lsize=4)


			#### 3 conv 第3层卷积 ####			
			## conv size变化为 7*7———> 7*7 ，ceil(7/1) ##
			conv3=conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
			
			## pool size变化为 7*7———> 4*4 ，ceil(7/2) ##
			pool3=maxpool2d('pool3',conv3,k=2)
			
			## norm size变化为 4*4———> 4*4 ##
			norm3=norm('norm3', pool3, lsize=4)



			#### 4 conv 第4层卷积####			
			## conv size变化为 4*4———> 4*4 ，ceil(4/1) ##
			conv4=conv2d('conv4', norm3, weights['wc4'], biases['bc4'])


			#### 5 conv 第5层卷积####			
			## conv size变化为 4*4———> 4*4 ，ceil(4/1) ##
			conv5=conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
			
			## pool size变化为 4*4———> 2*2 ，ceil(4/2) ##
			pool5=maxpool2d('pool5',conv5,k=2)
			
			## norm size变化为 2*2———> 2*2 ##
			norm5=norm('norm5', pool5, lsize=4)


			#### 1 fc 第1全连接层 'wd1': tf.Variable(tf.random_normal([2*2*256,4096]))，因此wd1的shape为2*2*256而不是4*4*256（原书有些问题）####			
			fc1=tf.reshape(norm5,[-1,weights['wd1'].get_shape().as_list()[0]])
			fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
			fc1=tf.nn.relu(fc1)


			## dropout 丢弃层 ##			
			fc1=tf.nn.dropout(fc1, dropout)


			#### 2 fc 第2全连接层 ####			
			fc2=tf.reshape(fc1,[-1,weights['wd2'].get_shape().as_list()[0]])
			fc2=tf.add(tf.matmul(fc2,weights['wd2']),biases['bd2'])
			fc2=tf.nn.relu(fc2)

			## dropout 丢弃层 ##			
			fc2=tf.nn.dropout(fc2, dropout)


			#### output 输出层 ####			
			out=tf.add(tf.matmul(fc2,weights['out']),biases['out'])
			return out
