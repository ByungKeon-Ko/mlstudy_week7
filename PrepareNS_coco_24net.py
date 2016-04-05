
import Image
import numpy as np
import image_search
import tensorflow as tf
import cvpr_network
import time
import random

# ------ Parameters ---------------------- 
base_path		= "/home/bkko/ml_study/week7"
coco_path		= "%s/../coco" %base_path

TotalNum	= 5764
# threshold1 = 0.15
# threshold2 = 0.1
# threshold1 = 0.97
threshold1 = 0.05
threshold2 = 0.001

MinFace = 28

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2 )
ckpt_file1 = "tmp/model_12net_2.ckpt"
ckpt_file2 = "tmp/model_calib12_4.ckpt"
ckpt_file3 = "tmp/model_24net_1.ckpt"
ckpt_file4 = "tmp/model_calib24.ckpt"

# net_name = '24net'
net_name = '48net'

# ------ Network ------------------------
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

if (net_name == '48net') :
	N24 = cvpr_network.cvpr_24net()
	N24.input_config_noscale()
	N24.infer()
	
	CAL24 = cvpr_network.cvpr_calib_24net()
	CAL24.input_config_noscale()
	CAL24.infer()

	saver2 = tf.train.Saver( { 'net24_w_conv1':N24.W_conv1, 'net24_b_conv1':N24.b_conv1, 'net24_w_fc1':N24.W_fc1, 'net24_b_fc1':N24.b_fc1, 'net24_w_fc2':N24.W_fc2, 'net24_b_fc2': N24.b_fc2 } )
	saver2.restore(sess, ckpt_file3)
	saver3 = tf.train.Saver( { 'cal24_w_conv1':CAL24.W_conv1, 'cal24_b_conv1':CAL24.b_conv1, 'cal24_w_fc1':CAL24.W_fc1, 'cal24_b_fc1':CAL24.b_fc1, 'cal24_w_fc2':CAL24.W_fc2, 'cal24_b_fc2': CAL24.b_fc2 } )
	saver3.restore(sess, ckpt_file4)


if net_name == '24net' :
	network_list = [N12, CAL12]
elif net_name == '48net' :
	network_list = [N12, CAL12, N24, CAL24]
	# network_list = [N12, CAL12, N24]

# ------ functions ------------------------
def create_ns (tmp_imgpath, cnt_ns, network_list, threshold1, threshold2 ) :
	
	tmp_img = Image.open("%s/%s" %(coco_path, tmp_imgpath), 'r' )
	org_img = Image.open("%s/%s" %(coco_path, tmp_imgpath), 'r' )

	down_scale = 1
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
	size = tmp_img.size

	resize_ratio = float(MinFace)/ 12. * down_scale
	try :
		tmp_img = tmp_img.resize( (int(size[0]/resize_ratio), int(size[1]/resize_ratio)), Image.BILINEAR )
	except IOError :
		sys.exit("truncated byte error!")

	false_pos_annot = image_search.detect_pos(network_list, tmp_img, threshold1, threshold2, 'aflw', resize_ratio )

	if len(false_pos_annot) <= 0 :
		return 0

	# if type(false_pos_annot)==list :
	# 	false_pos_annot = image_search.apply_nms( false_pos_annot )
	# else :
	# 	false_pos_annot = image_search.apply_nms( false_pos_annot.tolist() )

	# image_search.save_annot(false_pos_annot)
	cnt_save = 0
	for j in xrange( len(false_pos_annot) ) :
		if net_name == '24net' :
			path = "%s/NS_det24_2/ns-%d.jpg" %(base_path, cnt_ns+cnt_save)
			org_img = org_img.convert('RGB')
			image_search.save_patch(org_img, false_pos_annot[j], path, net_name)
			cnt_save = cnt_save + 1

		elif net_name == '48net' :
			path = "%s/NS_det48_2/ns-%d.jpg" %(base_path, cnt_ns+cnt_save)
			org_img = org_img.convert('RGB')
			image_search.save_patch(org_img, false_pos_annot[j], path, net_name)
			cnt_save = cnt_save + 1

	return cnt_save

# ----------------------------------------- 
cnt_ns			= 0
cnt_img			= 0
tmp_imgpath		= 0
rect_file_path	= "%s/file_list"	  %(coco_path)
rect_file		= open(rect_file_path,  'r')

time1 = time.time()
print "generating NS start for ", net_name
for i in range(1,TotalNum+1) :
	tmp_imgpath = rect_file.readline().rstrip()
	cnt_img = cnt_img + 1
	cnt_ns = cnt_ns + create_ns(tmp_imgpath, cnt_ns, network_list, threshold1, threshold2 )

	if i%10 == 0 :
		time2 = time1
		time1 = time.time()
		# print "Calc Time : ", time1 - time2
		print "current step : ", cnt_ns, cnt_img

	if cnt_ns > 200000 :
		print "Over 20K! Finish!"
		break

print "Loop expired!! Finish! @ image = ", i


