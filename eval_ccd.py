#
#
#

import Image
import numpy as np
import tensorflow as tf
import random
import math
import scipy.ndimage
import time
from skimage.transform import pyramid_gaussian

import cvpr_network

# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/week7"
test_list = "%s/../fddb/FDDB-folds/rect-9.txt" %base_path

threshold1 = 0.80
threshold2 = 0.90
threshold3 = 0.95
BigKernelSize = 100
MinFace = 48

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1 )

# ----------------------------------------- 

# ------ Network ------------------------
sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

N12 = cvpr_network.cvpr_12net()
N12.infer_scan(BigKernelSize)

N12_infer = cvpr_network.cvpr_12net()
N12_infer.infer()

N24 = cvpr_network.cvpr_24net()
# N24.infer_scan(BigKernelSize)
N24.infer()

N48 = cvpr_network.cvpr_48net()
# N48.infer_scan(BigKernelSize)
N48.infer()

# ------ Scanning ------------------------
saver1 = tf.train.Saver( { 'net48_w_conv1':N48.W_conv1, 'net48_b_conv1':N48.b_conv1, 'net48_w_conv2':N48.W_conv2, 'net48_b_conv2':N48.b_conv2, 'net48_w_fc1':N48.W_fc1, 'net48_b_fc1':N48.b_fc1, 'net48_w_fc2':N48.W_fc2, 'net48_b_fc2': N48.b_fc2 } )
saver2 = tf.train.Saver( { 'net24_w_conv1':N24.W_conv1, 'net24_b_conv1':N24.b_conv1, 'net24_w_fc1':N24.W_fc1, 'net24_b_fc1':N24.b_fc1, 'net24_w_fc2':N24.W_fc2, 'net24_b_fc2': N24.b_fc2 } )
saver3 = tf.train.Saver( { 'net12_w_conv1':N12.W_conv1, 'net12_b_conv1':N12.b_conv1, 'net12_w_fc1':N12.W_fc1, 'net12_b_fc1':N12.b_fc1, 'net12_w_fc2':N12.W_fc2, 'net12_b_fc2': N12.b_fc2 } )
saver4 = tf.train.Saver( { 'net12_w_conv1':N12_infer.W_conv1, 'net12_b_conv1':N12_infer.b_conv1, 'net12_w_fc1':N12_infer.W_fc1, 'net12_b_fc1':N12_infer.b_fc1, 'net12_w_fc2':N12_infer.W_fc2, 'net12_b_fc2': N12_infer.b_fc2 } )

saver1.restore(sess, "tmp/model_48net.ckpt")
saver2.restore(sess, "tmp/model_24net.ckpt")
saver3.restore(sess, "tmp/model_12net.ckpt")
saver4.restore(sess, "tmp/model_12net.ckpt")
print "Model restored"

rect_file = open(test_list,  'r')

while 1 :
	temp_line = rect_file.readline().rstrip()
	
	cond_eof = temp_line == ''
	cond_numface = len(temp_line) <= 2
	if cond_numface != 1 :
		cond_newimg = (temp_line[0:3] == '200') & (temp_line[3]!=' ') & (temp_line[3]!='.')
	else :
		cond_newimg = 0

	if cond_eof :	   # end of file
		print "1 set done!!"
		break
	
	elif cond_numface : # the number of face in the image
		pass
	
	elif cond_newimg :  # new image name
		org_img_file = Image.open("%s/../fddb/%s.jpg" %(base_path, temp_line) )
		
		start_time = time.time()
		pyramids = list( pyramid_gaussian(org_img_file, downscale=math.sqrt(2) ) )
		
		for i in range(len(pyramids) ):
			if min( pyramids[i].shape[0], pyramids[i].shape[1] ) < MinFace :
			# if min( pyramids[i].shape[0], pyramids[i].shape[1] ) < BigKernelSize :
				del pyramids[i:]
				break
			
		cnt_face = 0
		
		contents = []
		# contents.append( test_file_name )
		contents.append( temp_line )
		
		batch = np.zeros([BigKernelSize,BigKernelSize,3])

		def next_batch(j, i, l) :
			global pos_y
			global pos_x
			global BigKernelSize
			for pos_y in range(BigKernelSize) :
				for pos_x in range(BigKernelSize) :
					if (j + pos_y >= pyramids[l].shape[0]) | (i + pos_x >= pyramids[l].shape[1]) :
						batch[pos_y, pos_x] = [0,0,0]
					else :
						batch[pos_y, pos_x] = pyramids[l][j + pos_y, i + pos_x]
		
			return batch
		
		for l in range( 2, len(pyramids) ) :
			temp_row = pyramids[l].shape[0]
			temp_col = pyramids[l].shape[1]
		
			print " ONE pyramid!! %d" %(l)
			j = 0
			i = 0

			while 1 :
				while 1 :
					# print " one loop!! %d, %d, %d" %(l,j, i)

					batch = next_batch(j,i, l)

					batch_reshape = np.reshape(batch, [1, BigKernelSize*BigKernelSize*3] )
					y_val_scan = sess.run(N12.y_conv, feed_dict={N12.x:batch_reshape } )
					y_val_scan = np.reshape(y_val_scan, [8, 8 ] )

					y_val_shape = np.shape(y_val_scan)

					for bigpatch_y in range(y_val_shape[0]):
						for bigpatch_x in range(y_val_shape[1]):
							if y_val_scan[bigpatch_y,bigpatch_x] >= threshold1 :
								tmp_row = (bigpatch_y*4)
								tmp_col = (bigpatch_x*4)

								batch_48 = np.reshape(batch[tmp_row:tmp_row+MinFace, tmp_col:tmp_col+MinFace] , [1, MinFace* MinFace * 3] )
								ccd_12net = N12_infer.h_fc1.eval( feed_dict={ N12_infer.x:batch_48 } )
								y_val = N24.y_conv.eval( feed_dict={N24.x:batch_48, N24.in_12net:ccd_12net } )

								if y_val >= threshold2 :
									ccd_24net_1 = N24.h_fc1.eval( feed_dict={ N24.x:batch_48, N24.in_12net:ccd_12net } )
									ccd_24net = np.concatenate( [ccd_12net, ccd_24net_1], axis = 3 )

									y_val = N48.y_conv.eval( feed_dict = {N48.x:batch_48, N48.in_24net:ccd_24net } )

									if y_val >= threshold3 :
										# print "correct!!"
										org_row = (j + bigpatch_y*4) * math.sqrt(2)**l
										org_col = (i + bigpatch_x*4) * math.sqrt(2)**l
										org_len = MinFace * math.sqrt(2)**l
										contents.append( "%d %d %d %d %f %d" %(org_col, org_row + org_len, org_len, org_len, y_val[0,0,0,0], l ) )
										cnt_face = cnt_face + 1
									else :
										pass
										# print "wrong!!"
		
					if i == ( temp_col - BigKernelSize ) :
						i = 0
						# print "escape XXXXXXX !!"
						break

					incre_x = BigKernelSize - int(1.5* MinFace) + 3
					if i + incre_x + BigKernelSize <= temp_col :
						i = i + incre_x
					elif temp_col - BigKernelSize < 0 :
						break
					else :
						i = temp_col - BigKernelSize

				if j == ( temp_row - BigKernelSize ) :
					j = 0
					# print "escape y !!"
					break

				incre_y = BigKernelSize - int(1.5 * MinFace) + 3
				if j + incre_y + BigKernelSize < temp_row :
					j = j + incre_y
				elif temp_row - BigKernelSize < 0 :
					break
				else :
					j = temp_row - BigKernelSize

		contents.insert(1, str(cnt_face) )
		
		print " ------ %s seconds ------ " %( time.time() - start_time )
		output_file = open("%s/ofile_48net/out-%s.txt" %(base_path, temp_line.split('/')[4]), 'w' )
		
		for i in range(cnt_face+2) :
			output_file.write("%s\n" %contents[i] )


