
import Image
import numpy as np
import math
import random
import tensorflow as tf

import cvpr_network
import image_search
import time

# ------ Parameters ---------------------- 
# dataset_name = 'aflw'
dataset_name = 'fddb'
func_mode = 'table'
# func_mode = 'annot'

base_path		= "/home/bkko/ml_study/week7"
if dataset_name == 'aflw' :
	annot_file_path  = "/home/bkko/ml_study/aflw/aflw/annotation/annot_bkko"
	aflw_path		= "/home/bkko/ml_study/aflw/aflw/data/flickr"
elif dataset_name == 'fddb' :
	annot_file_path  = "/home/bkko/ml_study/fddb/FDDB-folds/rectbtm-1.txt"
	aflw_path		= "/home/bkko/ml_study/fddb"

if func_mode == 'annot' :
	output_annot_path = "/home/bkko/ml_study/week7/ofile_12net/o_annot.txt"
	output_annot = open(output_annot_path, 'w')

# print output_annot_path
# print annot_file_path

if dataset_name == 'fddb' :
	MinFace = 28
elif dataset_name == 'aflw' :
	MinFace = 24

net_name = '12net'
# net_name = '24net'
# net_name = '48net'

if net_name == '12net' :
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05 )
elif net_name == '24net' :
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20 )
elif net_name == '48net' :
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15 )

# threshold1 = 0.3	# 96.2%

threshold1 = 0.05
# threshold1 = 0.025

threshold2 = 0.00001
# threshold2 = 1e-4
# threshold3 = 1e-30

# threshold2 = 0.9
# threshold3 = 0.0
threshold3 = 0.5

ckpt_file1 = "tmp/model_12net_2.ckpt"
# ckpt_file1 = "tmp/model_12net_3.ckpt"
ckpt_file2 = "tmp/model_calib12_4.ckpt"
# ckpt_file3 = "tmp/model_24net.ckpt"
ckpt_file3 = "tmp/model_24net_1.ckpt"
ckpt_file4 = "tmp/model_calib24.ckpt"
ckpt_file5 = "tmp/model_48net_1.ckpt"
# ckpt_file5 = "tmp_bck/model_48net_1.ckpt"
ckpt_file6 = "tmp/model_calib48.ckpt"

# ------ functions ------------------------
# def save_tmp_img( tmp_matrix, new_index ) :
# 	tmp_img = Image.fromarray( tmp_matrix )
# 	tmp_img.save("%s/NS_24net/ns-%s.jpg" %(base_path, new_index) )

# -----------------------------------------

sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

N12 = cvpr_network.cvpr_12net()
# N12.infer_scan()
N12.input_config_noscale()
N12.infer()

CAL12 = cvpr_network.cvpr_calib_12net()
CAL12.input_config_noscale()
CAL12.infer()

saver1 = tf.train.Saver( { 'net12_w_conv1':N12.W_conv1, 'net12_b_conv1':N12.b_conv1, 'net12_w_fc1':N12.W_fc1, 'net12_b_fc1':N12.b_fc1, 'net12_w_fc2':N12.W_fc2, 'net12_b_fc2': N12.b_fc2 } )
saver1.restore(sess, ckpt_file1)
saver2 = tf.train.Saver( { 'cal12_w_conv1':CAL12.W_conv1, 'cal12_b_conv1':CAL12.b_conv1, 'cal12_w_fc1':CAL12.W_fc1, 'cal12_b_fc1':CAL12.b_fc1, 'cal12_w_fc2':CAL12.W_fc2, 'cal12_b_fc2': CAL12.b_fc2 } )
saver2.restore(sess, ckpt_file2)

if (net_name == '24net') | (net_name == '48net') :
	N24 = cvpr_network.cvpr_24net()
	N24.input_config_noscale()
	N24.infer()
	
	CAL24 = cvpr_network.cvpr_calib_24net()
	CAL24.input_config_noscale()
	CAL24.infer()

	saver3 = tf.train.Saver( { 'net24_w_conv1':N24.W_conv1, 'net24_b_conv1':N24.b_conv1, 'net24_w_fc1':N24.W_fc1, 'net24_b_fc1':N24.b_fc1, 'net24_w_fc2':N24.W_fc2, 'net24_b_fc2': N24.b_fc2 } )
	saver3.restore(sess, ckpt_file3)
	saver4 = tf.train.Saver( { 'cal24_w_conv1':CAL24.W_conv1, 'cal24_b_conv1':CAL24.b_conv1, 'cal24_w_fc1':CAL24.W_fc1, 'cal24_b_fc1':CAL24.b_fc1, 'cal24_w_fc2':CAL24.W_fc2, 'cal24_b_fc2': CAL24.b_fc2 } )
	saver4.restore(sess, ckpt_file4)

if (net_name == '48net') :
	N48 = cvpr_network.cvpr_48net()
	N48.infer()
	
	CAL48 = cvpr_network.cvpr_calib_48net()
	CAL48.input_config_noscale()
	CAL48.infer()

	saver5 = tf.train.Saver( { 'net48_w_conv1':N48.W_conv1, 'net48_b_conv1':N48.b_conv1, 'net48_w_conv2':N48.W_conv2, 'net48_b_conv2':N48.b_conv2, 'net48_w_fc1':N48.W_fc1, 'net48_b_fc1':N48.b_fc1, 'net48_w_fc2':N48.W_fc2, 'net48_b_fc2': N48.b_fc2 } )
	saver5.restore(sess, ckpt_file5)
	saver6 = tf.train.Saver( { 'cal48_w_conv1':CAL48.W_conv1, 'cal48_b_conv1':CAL48.b_conv1, 'cal48_w_conv2':CAL48.W_conv2, 'cal48_b_conv2':CAL48.b_conv2, 'cal48_w_fc1':CAL48.W_fc1, 'cal48_b_fc1':CAL48.b_fc1, 'cal48_w_fc2':CAL48.W_fc2, 'cal48_b_fc2': CAL48.b_fc2 } )
	saver6.restore(sess, ckpt_file6)

if net_name == '12net' :
	# network_list = [N12, CAL12]
	network_list = [N12]
elif net_name == '24net' :
	network_list = [N12, CAL12, N24, CAL24]
	# network_list = [N12, CAL12, N24]
elif net_name == '48net' :
	network_list = [N12, CAL12, N24, CAL24, N48, CAL48]
	# network_list = [N12, CAL12, N24, CAL24, N48]

# -----------------------------------------
annot_file = open(annot_file_path, 'r' ).readlines()

tmp_img_path = 0
tmp_img = 0
tmp_numface = 0
tmp_annot_list = []
tmp_annot = 0

total_true_pos = 0
total_pos = 0
cond_eof = 0
cond_newimg = 0
cond_numface =0
cond_annot = 0
img_cnt = 0
tmp_num_window = 0
total_num_window = 0
start_flag = 1
annot_cnt = 0
tmp_annot = np.zeros([5]).astype(np.str)
annot_file_index = 0

start_time = time.time()

print " current dataset and func_mode are ", dataset_name, func_mode, net_name, len(network_list)

while 1 :
	tmp_line = annot_file[annot_file_index].rstrip()
	annot_file_index = annot_file_index + 1
	cond_newimg, cond_numface, cond_annot, cond_eof = image_search.decoding_annot(tmp_line ) 

	if cond_newimg :
		img_cnt = img_cnt + 1
		if start_flag != 1 :
			if truncatedbyte_flag == 0 :
				if func_mode == 'table' :
					tmp_true_pos, tmp_num_window = image_search.find_true_pos(network_list, tmp_img, tmp_numface, tmp_annot_list, threshold1, threshold2, threshold3, dataset_name, resize_ratio, func_mode )
					total_true_pos = total_true_pos + tmp_true_pos
					total_num_window = total_num_window + tmp_num_window
				elif func_mode == 'annot' :
					final_list = image_search.find_true_pos(network_list, tmp_img, tmp_numface, tmp_annot_list, threshold1, threshold2, threshold3, dataset_name, resize_ratio, func_mode )
					output_annot.write("%s\n"%tmp_img_path)
					output_annot.write("%s\n"%str(len(final_list) ) )
					for k in xrange( len(final_list) ) :
						output_annot.write("%s %s %s %s %s\n" %( int(final_list[k][0]), int(final_list[k][1]), int(final_list[k][2]), int(final_list[k][3]), final_list[k][4]) )
			else :
				output_annot.write("%s\n"%tmp_img_path)
				output_annot.write("0\n" )

		if (img_cnt%10) == 0 :
			# print "%s-th image" %img_cnt, "find face? : ", tmp_true_pos
			print "Current Image = %s, image_search time = %s" %(img_cnt, time.time() - start_time)
			recall_rate = float(total_true_pos) /total_pos
			# print "tmp recall rate on 12net : ", recall_rate
			# print "tmp window : ", float(total_num_window)/img_cnt
			start_time = time.time()

		start_flag = 0

		tmp_img_path = tmp_line.rstrip()
		if dataset_name == 'aflw' :
			tmp_img = Image.open("%s/%s"%(aflw_path, tmp_img_path) )
		elif dataset_name == 'fddb' :
			tmp_img = Image.open("%s/%s.jpg"%(aflw_path, tmp_img_path) )

		size = tmp_img.size
		down_scale = 1
		if (dataset_name == 'aflw') :
			if (max(tmp_img.size[0], tmp_img.size[1]) > 3000) :
				down_scale = 6
			elif (max(tmp_img.size[0], tmp_img.size[1]) > 2500) :
				down_scale = 5
			elif (max(tmp_img.size[0], tmp_img.size[1]) > 2000) :
				down_scale = 4
			elif (max(tmp_img.size[0], tmp_img.size[1]) > 1500) :
				down_scale = 3
			elif (max(tmp_img.size[0], tmp_img.size[1]) > 1000) :
				down_scale = 2
			elif (max(tmp_img.size[0], tmp_img.size[1]) > 750) :
				down_scale = 1.5
			# size = tmp_img.size

		resize_ratio = float(MinFace)/ 12. * down_scale
		truncatedbyte_flag = 0
		try :
			tmp_img = tmp_img.resize( (int(size[0]/resize_ratio), int(size[1]/resize_ratio)), Image.BILINEAR )
		except IOError :
			truncatedbyte_flag = 1

	elif cond_numface :
		tmp_numface = int( tmp_line.rstrip() )
		tmp_annot_list = np.zeros([tmp_numface,5]).astype(np.float32)
		annot_cnt = 0
		total_pos = total_pos + tmp_numface

	elif cond_annot :
		tmp_annot[0:4] = np.array( tmp_line.split()[0:4] ).astype(np.int16)
		tmp_annot[4] = 0		# flag
		# tmp_annot = np.array( tmp_annot ).astype(np.int16)
		# tmp_annot_list.append(tmp_annot)
		tmp_annot_list[annot_cnt] = tmp_annot
		annot_cnt = annot_cnt + 1

	elif cond_eof :
		print "cond eof!"
		if truncatedbyte_flag == 0 :
			if func_mode == 'table' :
				tmp_true_pos, tmp_num_window = image_search.find_true_pos( network_list, tmp_img, tmp_numface, tmp_annot_list, threshold1, threshold2, threshold3, dataset_name, resize_ratio, func_mode )
				total_true_pos = total_true_pos + tmp_true_pos
				total_num_window = total_num_window + tmp_num_window
			elif func_mode == 'annot' :
				final_list = image_search.find_true_pos(network_list, tmp_img, tmp_numface, tmp_annot_list, threshold1, threshold2, threshold3, dataset_name, resize_ratio, func_mode )
				output_annot.write("%s\n"%tmp_img_path)
				output_annot.write("%s\n"%str(len(final_list) ) )
				for k in xrange( len(final_list) ) :
					output_annot.write("%s %s %s %s %s\n" %( int(final_list[k][0]), int(final_list[k][1]), int(final_list[k][2]), int(final_list[k][3]), final_list[k][4]) )
		else :
			output_annot.write("%s\n"%tmp_img_path)
			output_annot.write("0\n" )

		break



recall_rate = float(total_true_pos) /total_pos
print "@@@@@@ Recall rate on 12net : ", recall_rate
print "total window : ", float(total_num_window)/img_cnt

