# Alexnet_MNIST
利用Alexnet实现MNIST
===================

利用Alexnet的5层卷积结构实现MNIST识别（注意：层数传递时影像的大小很重要，直接决定网络是否正确）
----------------------------------------------------------------------------------

对于conv和pool层，均涉及padding的类型，padding有两种类型，一种是'SAME'，一种是'VALID'。
-------------------------------------------------------------------------------

		对于SAME，the output height and width are computed as:

		out_height = ceil(float(in_height) / float(strides[1]))

		out_width = ceil(float(in_width) / float(strides[2]))


		对于VALID，the output height and width are computed as:

		out_height = ceil(float(in_height - filter_height + 1) / float(strides1))

		out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

		ceil表示向上取整

参考李嘉璇的tensorflow和网上的代码，总结如下：
----------------------------------------
----------------------------------------------

# 一.李嘉璇————定义5层网络结构，每层都是卷积+池化+规范化

## 1.卷积

		########## define conv process ##########
		def conv2d(name,x,W,b,strides=1):
			x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
			x=tf.nn.bias_add(x,b)
			return tf.nn.relu(x,name=name)

		########## define pool process ##########
		def maxpool2d(name,x,k=2):
			return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)

		########## define norm process ##########
		def norm(name, l_input, lsize=4):
			return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)

		def alex_net(x, weights, biases, dropout):
			#### reshape input picture 输入数字是1*784的数据，将其reshape成28*28的影像 ####
			x=tf.reshape(x, shape=[-1,28,28,1])

			#### 1 conv 第1层卷积 ####
			## conv size变化为 28*28————> ##
			conv1=conv2d('conv1', x, weights['wc1'], biases['bc1'])
			## pool ##
			pool1=maxpool2d('pool1',conv1,k=2)
			## norm ##
			norm1=norm('norm1', pool1, lsize=4)

			#### 2 conv ####
			## conv ##
			conv2=conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
			## pool ##
			pool2=maxpool2d('pool2',conv2,k=2)
			## norm ##
			norm2=norm('norm2', pool2, lsize=4)

			#### 3 conv ####
			## conv ##
			conv3=conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
			## pool ##
			pool3=maxpool2d('pool3',conv3,k=2)
			## norm ##
			norm3=norm('norm3', pool3, lsize=4)


			#### 4 conv ####
			## conv ##
			conv4=conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

			#### 5 conv ####
			## conv ##
			conv5=conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
			## pool ##
			pool5=maxpool2d('pool5',conv5,k=2)
			## norm ##
			norm5=norm('norm5', pool5, lsize=4)


			#### 1 fc ####
			fc1=tf.reshape(norm5,[-1,weights['wd1'].get_shape().as_list()[0]])
			fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
			fc1=tf.nn.relu(fc1)

			## dropout ##
			fc1=tf.nn.dropout(fc1, dropout)

			#### 2 fc ####
			fc2=tf.reshape(fc1,[-1,weights['wd2'].get_shape().as_list()[0]])
			fc2=tf.add(tf.matmul(fc2,weights['wd2']),biases['bd2'])
			fc2=tf.nn.relu(fc2)

			## dropout ##
			fc2=tf.nn.dropout(fc2, dropout)

			#### output ####
			out=tf.add(tf.matmul(fc2,weights['out']),biases['out'])
			return out
