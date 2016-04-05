
import Image
import numpy as np
import math
import random
import tensorflow as tf

import cvpr_network
import image_search
import time
from skimage.transform import pyramid_gaussian
import sys

# ------ Parameters ---------------------- 
dataset_name = 'aflw'
# func_mode = 'table'
func_mode = 'annot'

base_path		= "/home/bkko/ml_study/week7"
annot_file_path  = "/home/bkko/ml_study/aflw/aflw/annotation/annot_bkko"
aflw_path		= "/home/bkko/ml_study/aflw/aflw/data/flickr"

MinFace = 50
net_name = '48net'
# net_name = '24net'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4 )

threshold1 = 0.5
threshold2 = 0.5
ckpt_file1 = "tmp/model_12net_2.ckpt"
ckpt_file2 = "tmp/model_calib12_4.ckpt"
ckpt_file3 = "tmp/model_24net_1.ckpt"
ckpt_file4 = "tmp/model_calib24.ckpt"

# ------ functions ------------------------
def create_ns (tmp_imgpath, cnt_ns, network_list, threshold1) :
	
	tmp_img = Image.open("%s/%s" %(aflw_path, tmp_imgpath), 'r' )
	org_img = Image.open("%s/%s" %(aflw_path, tmp_imgpath), 'r' )

	size = tmp_img.size
	if max(size[0], size[1]) > 1000 :
		return 0

	resize_ratio = float(MinFace)/ 12.
	try :
		tmp_img = tmp_img.resize( (int(size[0]/resize_ratio), int(size[1]/resize_ratio)), Image.BILINEAR )
	except IOError :
		return 0

	scale_factor = 1.414
	pyra = list( pyramid_gaussian(tmp_img, downscale=scale_factor ) )
	max_pyra = image_search.remove_small_pyra( pyra )

	pos_list = []
	# Detection 12 net
	start_scale = 0
	for i in xrange( start_scale, max_pyra + 1 ) :
		try :
			pos_list.append( image_search.scan_12net(network_list[0], pyra[i], threshold1, i, scale_factor, 1) )
		except ValueError :
			pos_list.append( [ [-1,-1,-1,-1,-1] ])

	# Calibration 12 net
	for i in xrange(start_scale, max_pyra + 1 ) :
		scale_up = start_scale
		pos_list[i-start_scale] = image_search.apply_calib( network_list[1], pos_list[i-start_scale], pyra[i-scale_up], '12net', scale_factor, scale_up )

	# In-Scale NMS
	thre_cal1 = 0.5
	for i in xrange( start_scale, max_pyra + 1 ) :
		if type(pos_list[i-start_scale]) == list :
			pos_list[i-start_scale] = image_search.apply_nms( pos_list[i-start_scale], thre_cal1, 'inscale' )
		else :
			pos_list[i-start_scale] = image_search.apply_nms( pos_list[i-start_scale].tolist(), thre_cal1, 'inscale' )

	if net_name == '48net' :
		# Detection 24 net
		if len(network_list) >= 3 :
			for i in xrange(start_scale, max_pyra + 1 ) :
				scale_up = i
				pos_list[i] = image_search.apply_det( network_list, pos_list[i-start_scale], pyra[i-scale_up], threshold2, '24net', scale_factor, scale_up)

	# ADJUST SCALE and MERGE
	for i in xrange( start_scale, max_pyra + 1 ) :
		pos_list[i-start_scale] = np.array(pos_list[i-start_scale])
		pos_list[i-start_scale][:,0:4] = np.multiply( pos_list[i-start_scale][:,0:4], resize_ratio * scale_factor**i )

	final_list = []
	for i in xrange( start_scale, max_pyra + 1 ) :
		if pos_list[i-start_scale][0][2] > 0 :		# means there's face in this pyramid
			if len(final_list) == 0 :
				final_list = pos_list[i-start_scale]
			else :
				final_list = np.concatenate( [ final_list, pos_list[i-start_scale] ], axis = 0 )

	cnt_save = 0
	for i in xrange(len(final_list) ) :
		iou_thre = 0.1
		if image_search.check_iou(final_list[i], tmp_annot_list, iou_thre, 1) == 0 :
			if net_name == '24net' :
				path = "%s/NS_aflw24/ns-%d.jpg" %(base_path, cnt_ns+cnt_save)
			elif net_name == '48net' :
				path = "%s/NS_aflw48_1/ns-%d.jpg" %(base_path, cnt_ns+cnt_save)
			org_img = org_img.convert('RGB')
			image_search.save_patch(org_img, final_list[i], path, net_name)
			cnt_save = cnt_save + 1

	return cnt_save

# -----------------------------------------

sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

N12 = cvpr_network.cvpr_12net()
N12.input_config_noscale()
N12.infer()

CAL12 = cvpr_network.cvpr_calib_12net()
CAL12.input_config_noscale()
CAL12.infer()

saver1 = tf.train.Saver( { 'net12_w_conv1':N12.W_conv1, 'net12_b_conv1':N12.b_conv1, 'net12_w_fc1':N12.W_fc1, 'net12_b_fc1':N12.b_fc1, 'net12_w_fc2':N12.W_fc2, 'net12_b_fc2': N12.b_fc2 } )
saver1.restore(sess, ckpt_file1)
saver2 = tf.train.Saver( { 'cal12_w_conv1':CAL12.W_conv1, 'cal12_b_conv1':CAL12.b_conv1, 'cal12_w_fc1':CAL12.W_fc1, 'cal12_b_fc1':CAL12.b_fc1, 'cal12_w_fc2':CAL12.W_fc2, 'cal12_b_fc2': CAL12.b_fc2 } )
saver2.restore(sess, ckpt_file2)

if (net_name == '48net') :
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

if net_name == '24net' :
	network_list = [N12, CAL12]
elif net_name == '48net' :
	network_list = [N12, CAL12, N24, CAL24]

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

start_time = time.time()

# cnt_ns			= 0
# annot_file_index = 0
cnt_ns			= 26000
annot_file_index = 1420

print "Generating NS from aflw ", dataset_name, func_mode, net_name

while 1 :
	tmp_line = annot_file[annot_file_index].rstrip()
	annot_file_index = annot_file_index + 1
	cond_newimg, cond_numface, cond_annot, cond_eof = image_search.decoding_annot(tmp_line ) 

	if cond_newimg :
		img_cnt = img_cnt + 1
		tmp_imgpath = tmp_line.rstrip()

	elif cond_numface :
		tmp_numface = int( tmp_line.rstrip() )
		tmp_annot_list = np.zeros([tmp_numface,5]).astype(np.float32)
		annot_cnt = 0
		total_pos = total_pos + tmp_numface

	elif cond_annot :
		tmp_annot[0:4] = np.array( tmp_line.split()[0:4] ).astype(np.int16)
		tmp_annot[4] = 0		# flag
		tmp_annot_list[annot_cnt] = tmp_annot
		annot_cnt = annot_cnt + 1
		cnt_ns = cnt_ns + create_ns(tmp_imgpath, cnt_ns, network_list, threshold1 )

		if img_cnt % 10 == 0 :
			print "current step : ", cnt_ns, img_cnt, (time.time() -start_time)/60.
			start_time = time.time()

		if img_cnt > 100000 :
			break

	elif cond_eof :
		break

print "finish : ", cnt_ns, img_cnt

