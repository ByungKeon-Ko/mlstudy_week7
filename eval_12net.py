#
#
#

import Image
import numpy as np
import tensorflow as tf
import random
import math
from skimage.transform import pyramid_gaussian
import time

import cvpr_network

# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/week7"
test_list = "%s/../fddb/FDDB-folds/rect-9.txt" %base_path

threshold = 0.70
# BigKernelSize = 100
# BigKernelSize = 300
# BigKernelSize = 96
BigKernelSize = 144
# BigKernelSize = 192
# BigKernelSize = 288
# BigKernelSize = 480
MinFace = 48

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8 )
ckpt_file = "tmp/model_12net.ckpt"

# ----------------------------------------- 

# ------ Network ------------------------
sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

start_time = time.time()
N12 = cvpr_network.cvpr_12net()
N12.infer_scan(BigKernelSize)

# ------ Scanning ------------------------

saver = tf.train.Saver()

saver.restore(sess, ckpt_file)
print "Model restored"

print " ------ Network Define : %s seconds ------ " %( time.time() - start_time )

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

		def next_batch(j, i) :
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
		
		# for l in range( 2, len(pyramids) ) :
		for l in range( 2, 3 ) :
			temp_row = pyramids[l].shape[0]
			temp_col = pyramids[l].shape[1]
		
			print " ONE pyramid!! %d" %(l)
			j = 0
			i = 0

			while 1 :
				while 1 :
					print " one loop!! %d, %d, %d" %(l,j, i)

					batch = next_batch(j,i)

					batch_reshape = np.reshape(batch, [1, BigKernelSize*BigKernelSize*3] )
					y_val = sess.run(N12.y_conv, feed_dict={N12.x:batch_reshape } )
					# y_val = np.reshape(y_val, [8, 8 ] )
					# y_val = np.reshape(y_val, [7, 7 ] )
					y_val = np.reshape(y_val, [13, 13 ] )
					# y_val = np.reshape(y_val, [19, 19 ] )
					# y_val = np.reshape(y_val, [31, 31 ] )
					# y_val = np.reshape(y_val, [55, 55 ] )

					# y_val = np.reshape(y_val, [33, 33 ] )
					# y_val = np.reshape(y_val, [1, 1 ] )
					# y_val = np.reshape(y_val, [BigKernelSize/4, BigKernelSize/4 ] )

					y_val_shape = np.shape(y_val)

					for bigpatch_y in range(y_val_shape[0]):
						for bigpatch_x in range(y_val_shape[1]):
							# if l > 5 :
							# 	print l, bigpatch_x, bigpatch_y, y_val[bigpatch_y,bigpatch_x]
							if y_val[bigpatch_y,bigpatch_x] >= threshold :
								# print "correct!!"
								org_row = (j + bigpatch_y*4) * math.sqrt(2)**l
								org_col = (i + bigpatch_x*4) * math.sqrt(2)**l
								org_len = MinFace * math.sqrt(2)**l
								contents.append( "%d %d %d %d %f %d" %(org_col, org_row + org_len, org_len, org_len, y_val[bigpatch_y, bigpatch_x], l ) )
								cnt_face = cnt_face + 1
							else :
								pass
								# print "wrong!!"
		
					if i == ( temp_col - BigKernelSize ) :
						i = 0
						# print "escape XXXXXXX !!"
						break

					incre_x = BigKernelSize - int(1.1* MinFace) + 3
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

				incre_y = BigKernelSize - int(1.1 * MinFace) + 3
				if j + incre_y + BigKernelSize < temp_row :
					j = j + incre_y
				elif temp_row - BigKernelSize < 0 :
					break
				else :
					j = temp_row - BigKernelSize

		contents.insert(1, str(cnt_face) )
		
		print " ------ %s seconds ------ " %( time.time() - start_time )
		output_file = open("%s/ofile_12net/out-%s.txt" %(base_path, temp_line.split('/')[4]), 'w' )
		
		for i in range(cnt_face+2) :
			output_file.write("%s\n" %contents[i] )


