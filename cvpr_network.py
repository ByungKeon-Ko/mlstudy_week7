import tensorflow as tf

# ------ Parameters ---------------------- 
MinFace = 48
colorNum = 3

# ----------------------------------------

def weight_variable(shape, name):
	# initial = tf.random_normal(shape, stddev=0.01, name='initial')
	initial = tf.truncated_normal(shape, stddev=0.1, name='initial')
	return tf.Variable(initial, name = name)

def bias_variable(shape, name):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x,W) :
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID')

def max_pool_3x3(x):
	return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

class cvpr_12net () :
	def input_config_div2 (self):
		self.x		= tf.placeholder(tf.float32, [None, 24*24*3], name = 'x' )
		self.x_image_1	= tf.reshape(self.x, [-1,24,24, colorNum], name='x_image')
		self.x_image	= tf.image.resize_bilinear(self.x_image_1, [12,12])

	def input_config_div4 (self):
		self.x		= tf.placeholder(tf.float32, [None, 48*48*3], name = 'x' )
		self.x_image_1	= tf.reshape(self.x, [-1,48,48, colorNum], name='x_image')
		self.x_image	= tf.image.resize_bilinear(self.x_image_1, [12,12])

	def input_config_noscale (self):
		net_width = 12

		self.x			= tf.placeholder(tf.float32, [None, net_width*net_width*3], name = 'x' )
		self.x_image	= tf.reshape(self.x, [-1,net_width,net_width, colorNum], name='x_image')

	def infer (self):
		# ----- 1st Convolutional Layer --------- #
		# input size  : 48x48 x 3  channel ( 2D : nBatch, mixed )
		# output size : 14x14 x ?? channel ( 4D : nBatch, col, row, feature )
		ksize1	= 3
		
		self.W_conv1	= weight_variable([ksize1, ksize1, colorNum, 16], 'net12_w_conv1' )
		self.b_conv1	= bias_variable([16], 'net12_b_conv1' )
		self.h_conv1	= tf.nn.relu( conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1	= max_pool_3x3(self.h_conv1)

		# ----- 2nd FC layer --------------------- #
		# input : ? x ? x ? ch
		# output : 16 ch
		ksize2	= 4
		
		self.W_fc1	= weight_variable([ksize2, ksize2, 16, 16], 'net12_w_fc1' )
		self.b_fc1	= bias_variable([16], 'net12_b_fc1')
		self.h_fc1	= tf.nn.relu( conv2d(self.h_pool1, self.W_fc1) + self.b_fc1 )

		# --- 3rd FC Layer : Readout Layer --------------------- #
		# input  : ? x ? x 16 channel
		# output : 1 channel
		ksize3	= 1
		self.W_fc2	= weight_variable([ksize3, ksize3, 16, 1], 'net12_w_fc2' )
		self.b_fc2	= bias_variable([1], 'net12_b_fc2')
		self.ttemp	= conv2d(self.h_fc1, self.W_fc2) + self.b_fc2
		self.y_conv	= tf.sigmoid( conv2d(self.h_fc1, self.W_fc2) + self.b_fc2 )

	def objective (self):
		self.y_	= tf.placeholder(tf.float32, [None , 1], name	= 'y_' )
		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 1] )
		self.cross_entropy	= -tf.reduce_mean(self.y_ *tf.log(self.y_conv_reshape+1e-25) + (1-self.y_ ) *tf.log(1-self.y_conv_reshape+1e-25) )

	def train (self, LearningRate, threshold ):
		# self.train_step	= tf.train.AdamOptimizer(LearningRate).minimize(self.cross_entropy)
		# self.train_step	= tf.train.AdagradOptimizer(LearningRate).minimize(self.cross_entropy)
		self.train_step	= tf.train.GradientDescentOptimizer(LearningRate).minimize(self.cross_entropy)

		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 1] )
		tmp_node	= tf.cast( tf.greater( self.y_conv_reshape, threshold ), tf.float32)
		self.correct_prediction	= tf.equal( tmp_node , self.y_  )
		self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

	def infer_scan (self):
		self.x		= tf.placeholder(tf.float32 )

		# ----- 1st Convolutional Layer --------- #
		# input size  : 48x48 x 3  channel ( 2D : nBatch, mixed )
		# output size : 14x14 x ?? channel ( 4D : nBatch, col, row, feature )
		ksize1	= 3
		
		self.W_conv1	= weight_variable([ksize1, ksize1, colorNum, 16], 'net12_w_conv1' )
		self.b_conv1	= bias_variable([16], 'net12_b_conv1')
		self.h_conv1	= tf.nn.relu( conv2d(self.x, self.W_conv1) + self.b_conv1)
		self.h_pool1	= max_pool_3x3(self.h_conv1)

		# ----- 2nd FC layer --------------------- #
		# input : ? x ? x ? ch
		# output : 16 ch
		ksize2	= 4
		
		self.W_fc1	= weight_variable([ksize2, ksize2, 16, 16], 'net12_w_fc1' )
		self.b_fc1	= bias_variable([16], 'net12_b_fc1')
		self.h_fc1	= tf.nn.relu( conv2d(self.h_pool1, self.W_fc1) + self.b_fc1 )

		# --- 3rd FC Layer : Readout Layer --------------------- #
		# input  : ? x ? x 16 channel
		# output : 1 channel
		ksize3	= 1
		self.W_fc2	= weight_variable([ksize3, ksize3, 16, 1], 'net12_w_fc2' )
		self.b_fc2	= bias_variable([1], 'net12_b_fc2')
		self.ttemp	= conv2d(self.h_fc1, self.W_fc2) + self.b_fc2
		self.y_conv	= tf.sigmoid( conv2d(self.h_fc1, self.W_fc2) + self.b_fc2 )


class cvpr_24net () :
	def input_config (self):
		self.x		= tf.placeholder(tf.float32, [None, 48*48*3], name = 'x' )
		self.x_image_1	= tf.reshape(self.x, [-1,48,48, colorNum], name='x_image')
		self.x_image	= tf.image.resize_bilinear(self.x_image_1, [24,24])
	def input_config_noscale (self):
		self.x		= tf.placeholder(tf.float32, [None, 24*24*3], name = 'x' )
		self.x_image	= tf.reshape(self.x, [-1,24,24, colorNum], name='x_image')

	def infer (self):
		# ----- 1st Convolutional Layer --------- #
		# input size  : 48x48 x 3  channel ( 2D : nBatch, mixed )
		# output size : 14x14 x ?? channel ( 4D : nBatch, col, row, feature )
		ksize1	= 5
		
		self.W_conv1	= weight_variable([ksize1, ksize1, colorNum, 64], 'net24_w_conv1' )
		self.b_conv1	= bias_variable([64], 'net24_b_conv1')
		self.h_conv1	= tf.nn.relu( conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1	= max_pool_3x3(self.h_conv1)

		# ----- 2nd FC layer --------------------- #
		ksize2	= 9
		
		self.W_fc1	= weight_variable([ksize2, ksize2, 64, 128], 'net24_w_fc1' )
		self.b_fc1	= bias_variable([128], 'net24_b_fc1' )
		self.h_fc1	= tf.nn.relu( conv2d(self.h_pool1, self.W_fc1) + self.b_fc1 )

		# --- 3rd FC Layer : Readout Layer --------------------- #
		# input  : ? x ? x 16 channel + h_fc1 of 12net
		# output : 1 channel
		self.in_12net = tf.placeholder(tf.float32, [None, 1, 1, 16], name = 'in_12net' )
		self.merged_net = tf.concat(3,  [self.h_fc1, self.in_12net] )

		ksize3	= 1
		self.W_fc2	= weight_variable([ksize3, ksize3, 128+16, 1], 'net24_w_fc2' )
		self.b_fc2	= bias_variable([1], 'net24_b_fc2' )
		self.ttemp	= conv2d(self.merged_net, self.W_fc2) + self.b_fc2
		self.y_conv	= tf.sigmoid( self.ttemp )

	def objective (self):
		self.y_	= tf.placeholder(tf.float32, [None , 1], name	= 'y_' )
		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 1] )
		self.cross_entropy	= -tf.reduce_mean(self.y_ *tf.log(self.y_conv_reshape +1e-25) + (1-self.y_ ) *tf.log(1-self.y_conv_reshape +1e-25) )
		self.gradient = tf.gradients( self.cross_entropy, [self.W_conv1, self.b_conv1, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 ] )

	def train (self, LearningRate, threshold ):
		# self.train_step	= tf.train.AdamOptimizer(LearningRate).minimize(self.cross_entropy)
		# self.train_step	= tf.train.AdagradOptimizer(LearningRate).minimize(self.cross_entropy)
		self.train_step	= tf.train.GradientDescentOptimizer(LearningRate).minimize(self.cross_entropy)

		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 1] )
		tmp_node	= tf.cast( tf.greater( self.y_conv_reshape, threshold ), tf.float32)
		self.correct_prediction	= tf.equal( tmp_node , self.y_  )
		self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

class cvpr_48net () :
	def infer (self):
		self.x		= tf.placeholder(tf.float32, [None, MinFace*MinFace*3], name = 'x' )
		self.x_image	= tf.reshape(self.x, [-1,MinFace,MinFace, colorNum], name='x_image')

		# ----- 1st Convolutional Layer --------- #
		# input : nBatch x 48 x 48 x 3ch
		ksize1	= 5
		
		self.W_conv1	= weight_variable([ksize1, ksize1, colorNum, 64], 'net48_w_conv1' )
		self.b_conv1	= bias_variable([64], 'net48_b_conv1')
		self.h_conv1	= tf.nn.relu( conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1	= max_pool_3x3(self.h_conv1)

		# ----- 2nd Convolutional Layer --------- #
		# input : nBatch x 21 x 21 x 64
		ksize2	= 5
		
		self.W_conv2	= weight_variable([ksize2, ksize2, 64, 64], 'net48_w_conv2' )
		self.b_conv2	= bias_variable([64], 'net48_b_conv2' )
		self.h_conv2	= tf.nn.relu( conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2	= max_pool_3x3(self.h_conv2)

		# ----- 3rd FC layer --------------------- #
		# input : nBatch x 8 x 8 x 64
		ksize3	= 8
		
		self.W_fc1	= weight_variable([ksize3, ksize3, 64, 256], 'net48_w_fc1' )
		self.b_fc1	= bias_variable([256], 'net48_b_fc1' )
		self.h_fc1	= tf.nn.relu( conv2d(self.h_pool2, self.W_fc1) + self.b_fc1 )

		# --- 4th FC Layer : Readout Layer --------------------- #
		# input  : 1 x 1 x 256 channel + h_fc1 of 24net
		self.in_24net = tf.placeholder(tf.float32, [None, 1, 1, 128+16], name = 'in_24net' )
		self.merged_net = tf.concat(3,  [self.h_fc1, self.in_24net] )

		ksize4	= 1
		self.W_fc2	= weight_variable([ksize4, ksize4, 256+128+16, 1], 'net48_w_fc2' )
		self.b_fc2	= bias_variable([1], 'net48_b_fc2' )
		self.ttemp	= conv2d(self.merged_net, self.W_fc2) + self.b_fc2
		self.y_conv	= tf.sigmoid( conv2d(self.merged_net, self.W_fc2) + self.b_fc2 )

	def objective (self):
		self.y_	= tf.placeholder(tf.float32, [None , 1], name	= 'y_' )
		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 1] )
		self.cross_entropy	= -tf.reduce_mean(self.y_ *tf.log(self.y_conv_reshape+1e-25) + (1-self.y_ ) *tf.log(1-self.y_conv_reshape+1e-25) )
		# self.cross_entropy	= -tf.reduce_sum(self.y_ *tf.log(self.y_conv_reshape+1e-25) + (1-self.y_ ) *tf.log(1-self.y_conv_reshape+1e-25) )

	def train (self, LearningRate, threshold ):
		# self.train_step	= tf.train.AdamOptimizer(LearningRate).minimize(self.cross_entropy)
		# self.train_step	= tf.train.AdagradOptimizer(LearningRate).minimize(self.cross_entropy)
		self.train_step	= tf.train.GradientDescentOptimizer(LearningRate).minimize(self.cross_entropy)

		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 1] )
		tmp_node	= tf.cast( tf.greater( self.y_conv_reshape, threshold ), tf.float32)
		self.correct_prediction	= tf.equal( tmp_node , self.y_  )
		self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

class cvpr_calib_12net () :
	def input_config (self):
		self.x		= tf.placeholder(tf.float32, [None, MinFace*MinFace*3], name = 'x' )
		self.x_image_1	= tf.reshape(self.x, [-1,MinFace,MinFace, colorNum], name='x_image')
		self.x_image	= tf.image.resize_bilinear(self.x_image_1, [12,12])
	def input_config_noscale (self):
		self.x		= tf.placeholder(tf.float32, [None, 12*12*3], name = 'x' )
		self.x_image	= tf.reshape(self.x, [-1,12,12, colorNum], name='x_image')

	def infer (self):
		# ----- 1st Convolutional Layer --------- #
		ksize1	= 3
		
		self.W_conv1	= weight_variable([ksize1, ksize1, colorNum, 16], 'net12_w_conv1' )
		self.b_conv1	= bias_variable([16], 'net12_b_conv1' )
		self.h_conv1	= tf.nn.relu( conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1	= max_pool_3x3(self.h_conv1)

		# ----- 2nd FC layer --------------------- #
		ksize2	= 4
		
		self.W_fc1	= weight_variable([ksize2, ksize2, 16, 128], 'net12_w_fc1' )
		self.b_fc1	= bias_variable([128], 'net12_b_fc1')
		self.h_fc1	= tf.nn.relu( conv2d(self.h_pool1, self.W_fc1) + self.b_fc1 )

		# --- 3rd FC Layer : Readout Layer --------------------- #
		# output : 45 channel
		ksize3	= 1
		self.W_fc2	= weight_variable([ksize3, ksize3, 128, 45], 'net12_w_fc2' )
		self.b_fc2	= bias_variable([45], 'net12_b_fc2')
		self.conv3	= conv2d(self.h_fc1, self.W_fc2) + self.b_fc2
		self.conv3_reshape	= tf.reshape(self.conv3, [-1, 45])
		# self.y_conv	= tf.nn.softmax( conv2d(self.h_fc1, self.W_fc2) + self.b_fc2 )
		self.y_conv	= tf.nn.softmax( self.conv3_reshape )

	def objective (self):
		self.y_	= tf.placeholder(tf.float32, [None , 45], name	= 'y_' )
		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 45] )
		self.cross_entropy	= -tf.reduce_mean(self.y_ *tf.log(self.y_conv_reshape+1e-20) )

	def train (self, LearningRate ):
		self.train_step	= tf.train.GradientDescentOptimizer(LearningRate).minimize(self.cross_entropy)

		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 45] )
		self.y_select = tf.argmax(self.y_conv_reshape, 1)
		self.correct_prediction	= tf.equal( self.y_select , tf.argmax(self.y_, 1)  )
		self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

class cvpr_calib_24net () :
	def input_config (self):
		self.x		= tf.placeholder(tf.float32, [None, MinFace*MinFace*3], name = 'x' )
		self.x_image_1	= tf.reshape(self.x, [-1,MinFace,MinFace, colorNum], name='x_image')
		self.x_image	= tf.image.resize_bilinear(self.x_image_1, [24,24])

	def input_config_noscale (self):
		self.x		= tf.placeholder(tf.float32, [None, 24*24*3], name = 'x' )
		self.x_image	= tf.reshape(self.x, [-1,24,24, colorNum], name='x_image')

	def infer (self):
		# ----- 1st Convolutional Layer --------- #
		ksize1	= 5
		
		self.W_conv1	= weight_variable([ksize1, ksize1, colorNum, 32], 'net24_w_conv1' )
		self.b_conv1	= bias_variable([32], 'net24_b_conv1' )
		self.h_conv1	= tf.nn.relu( conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1	= max_pool_3x3(self.h_conv1)

		# ----- 2nd FC layer --------------------- #
		ksize2	= 9
		
		self.W_fc1	= weight_variable([ksize2, ksize2, 32, 64], 'net24_w_fc1' )
		self.b_fc1	= bias_variable([64], 'net24_b_fc1')
		self.h_fc1	= tf.nn.relu( conv2d(self.h_pool1, self.W_fc1) + self.b_fc1 )

		# --- 3rd FC Layer : Readout Layer --------------------- #
		# output : 45 channel
		ksize3	= 1
		self.W_fc2	= weight_variable([ksize3, ksize3, 64, 45], 'net24_w_fc2' )
		self.b_fc2	= bias_variable([45], 'net24_b_fc2')
		self.conv3	= conv2d(self.h_fc1, self.W_fc2) + self.b_fc2
		self.conv3_reshape	= tf.reshape(self.conv3, [-1, 45])
		self.y_conv	= tf.nn.softmax( self.conv3_reshape )

	def objective (self):
		self.y_	= tf.placeholder(tf.float32, [None , 45], name	= 'y_' )
		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 45] )
		self.cross_entropy	= -tf.reduce_mean(self.y_ *tf.log(self.y_conv_reshape+1e-20) )

	def train (self, LearningRate ):
		self.train_step	= tf.train.GradientDescentOptimizer(LearningRate).minimize(self.cross_entropy)

		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 45] )
		self.y_select = tf.argmax(self.y_conv_reshape, 1)
		self.correct_prediction	= tf.equal( self.y_select , tf.argmax(self.y_, 1)  )
		self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

class cvpr_calib_48net () :
	def input_config_noscale (self):
		self.x		= tf.placeholder(tf.float32, [None, 48*48*3], name = 'x' )
		self.x_image	= tf.reshape(self.x, [-1,48,48, colorNum], name='x_image')

	def infer (self):
		# ----- 1st Convolutional Layer --------- #
		ksize1	= 5
		
		self.W_conv1	= weight_variable([ksize1, ksize1, colorNum, 64], 'net48_w_conv1' )
		self.b_conv1	= bias_variable([64], 'net48_b_conv1' )
		self.h_conv1	= tf.nn.relu( conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1	= max_pool_3x3(self.h_conv1)

		# ----- 2nd Convolutional Layer --------- #
		ksize2	= 5
		
		self.W_conv2	= weight_variable([ksize2, ksize2, 64, 64], 'net48_w_conv2' )
		self.b_conv2	= bias_variable([64], 'net48_b_conv2' )
		self.h_conv2	= tf.nn.relu( conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

		# ----- 3rd FC layer --------------------- #
		ksize3	= 17
		
		self.W_fc1	= weight_variable([ksize3, ksize3, 64, 256], 'net48_w_fc1' )
		self.b_fc1	= bias_variable([256], 'net48_b_fc1')
		self.h_fc1	= tf.nn.relu( conv2d(self.h_conv2, self.W_fc1) + self.b_fc1 )

		# --- 4th FC Layer : Readout Layer --------------------- #
		# output : 45 channel
		ksize4	= 1
		self.W_fc2	= weight_variable([ksize4, ksize4, 256, 45], 'net48_w_fc2' )
		self.b_fc2	= bias_variable([45], 'net48_b_fc2')
		self.conv3	= conv2d(self.h_fc1, self.W_fc2) + self.b_fc2
		self.conv3_reshape	= tf.reshape(self.conv3, [-1, 45])
		self.y_conv	= tf.nn.softmax( self.conv3_reshape )

	def objective (self):
		self.y_	= tf.placeholder(tf.float32, [None , 45], name	= 'y_' )
		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 45] )
		self.cross_entropy	= -tf.reduce_mean(self.y_ *tf.log(self.y_conv_reshape+1e-20) )
		# self.cross_entropy	= -tf.reduce_sum(self.y_ *tf.log(self.y_conv_reshape+1e-20) )

	def train (self, LearningRate ):
		self.train_step	= tf.train.GradientDescentOptimizer(LearningRate).minimize(self.cross_entropy)

		self.y_conv_reshape	= tf.reshape( self.y_conv, [-1, 45] )
		self.y_select = tf.argmax(self.y_conv_reshape, 1)
		self.correct_prediction	= tf.equal( self.y_select , tf.argmax(self.y_, 1)  )
		self.accuracy	= tf.reduce_mean(tf.cast(self.correct_prediction, "float" ) )

