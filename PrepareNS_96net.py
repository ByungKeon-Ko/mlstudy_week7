
import Image
import numpy as np
import math
import random
import tensorflow as tf

import cvpr_network

# ------ Parameters ---------------------- 
base_path		= "/home/bkko/ml_study/week7"

threshold = 0.50

fold_num = 8
MinFace = 48
psNum = 4135
nsNum = 200

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1 )

# ------ functions ------------------------
def save_tmp_img( tmp_matrix, new_index ) :
	tmp_img = Image.fromarray( tmp_matrix )
	tmp_img.save("%s/NS_96net/ns-%s.jpg" %(base_path, new_index) )

# -----------------------------------------

sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

N12 = cvpr_network.cvpr_12net()
N12.infer()

N24 = cvpr_network.cvpr_24net()
N24.infer()

N48 = cvpr_network.cvpr_48net()
N48.infer()

saver_12net = tf.train.Saver( [N12.W_conv1, N12.b_conv1, N12.W_fc1, N12.b_fc1, N12.W_fc2, N12.b_fc2 ] )
saver_12net.restore(sess, "tmp/model_12net.ckpt")
saver_24net = tf.train.Saver( [N24.W_conv1, N24.b_conv1, N24.W_fc1, N24.b_fc1, N24.W_fc2, N24.b_fc2 ] )
saver_24net.restore(sess, "tmp/model_24net.ckpt")
saver_48net = tf.train.Saver( [N48.W_conv1, N48.b_conv1, N48.W_conv2, N48.b_conv2, N48.W_fc1, N48.b_fc1, N48.W_fc2, N48.b_fc2 ] )
saver_48net.restore(sess, "tmp/model_48net.ckpt")

new_index = 0

for i in xrange(psNum) :
	tmp_matrix = np.array( Image.open("%s/PS/ps-%d.jpg" %(base_path, i), 'r' ).getdata() ).reshape(MinFace, MinFace, 3 )
	tmp_matrix_reshape = np.reshape( np.divide(tmp_matrix, 255.0) , [1, MinFace*MinFace*3] )

	ccd_12net = N12.h_fc1.eval( feed_dict={ N12.x:tmp_matrix_reshape } )
	ccd_24net_1 = N24.h_fc1.eval( feed_dict={ N24.x:tmp_matrix_reshape, N24.in_12net:ccd_12net } )
	ccd_24net = np.concatenate( [ccd_12net, ccd_24net_1], axis = 3 )

	y_val = N48.y_conv.eval( feed_dict = {N48.x:tmp_matrix_reshape, N48.in_24net:ccd_24net } )
	y_val = np.reshape( y_val, [])

	if y_val >= threshold :
		pass
	else :
		new_index = new_index + 1

recall_rate = 1.0 - ( float(new_index) /psNum )
print "@@@@@@ Recall rate on 48net : ", recall_rate

new_index = 0
for i in xrange(nsNum) :
	tmp_matrix = np.array( Image.open("%s/NS_48net/ns-%d.jpg" %(base_path, i), 'r' ).getdata() ).reshape(MinFace, MinFace, 3 )
	tmp_matrix_reshape = np.reshape( np.divide(tmp_matrix, 255.0) , [1, MinFace*MinFace*3] )

	ccd_12net = N12.h_fc1.eval( feed_dict={ N12.x:tmp_matrix_reshape } )
	ccd_24net_1 = N24.h_fc1.eval( feed_dict={ N24.x:tmp_matrix_reshape, N24.in_12net:ccd_12net } )
	ccd_24net = np.concatenate( [ccd_12net, ccd_24net_1], axis = 3 )

	y_val = N48.y_conv.eval( feed_dict = {N48.x:tmp_matrix_reshape, N48.in_24net:ccd_24net } )
	y_val = np.reshape( y_val, [])

	if y_val >= threshold :
		save_tmp_img(tmp_matrix.astype(np.uint8), new_index)
		new_index = new_index + 1
	else :
		# print "Negative!!"
		pass

# False positive rate
fallout_rate = float(new_index)/nsNum

print "@@@@@@ Fall-out rate on 48net : ", fallout_rate

