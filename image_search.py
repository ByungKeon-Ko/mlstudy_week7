
from skimage.transform import pyramid_gaussian
import math
import numpy as np
import scipy
import CalibLib
from CalibLib import mapping_calib_para
from CalibLib import decoding_calib
from CalibLib import convert_invcalib
from CalibLib import transform_calib
import sys
import Image
import time
import copy

# MinFace = 48
mode_table = 'sliding'
# mode_table = '12net'

def find_true_pos (network_list, tmp_img, tmp_numface, tmp_annot_list, threshold1, threshold2, threshold3, dataset_name, resize_ratio, func_mode ) :

	start_scale = 0
	# start_scale = 8

	if dataset_name == 'aflw' :
		scale_factor = 1.414
	elif dataset_name == 'fddb' :
		scale_factor = 1.18

	pyra = list( pyramid_gaussian(tmp_img, downscale=scale_factor ) )
	max_pyra = remove_small_pyra( pyra )
	# max_pyra = start_scale+1

	if len(pyra[0].shape)!=3 :
		tmp_img = tmp_img.convert('RGB')
		pyra = list( pyramid_gaussian(tmp_img, downscale=scale_factor ) )
		max_pyra = remove_small_pyra( pyra )
	
	pos_list = []
	# Detection 12 net
	for i in xrange( start_scale, max_pyra + 1 ) :
		pos_list.append( scan_12net(network_list[0], pyra[i], threshold1, i, scale_factor, 0) )

	if len(network_list) >= 2 :
		# Calibration 12 net
		for i in xrange(start_scale, max_pyra + 1 ) :
			scale_up = start_scale
			pos_list[i-start_scale] = apply_calib( network_list[1], pos_list[i-start_scale], pyra[i-scale_up], '12net', scale_factor, scale_up )

		# In-Scale NMS
		thre_cal1 = 0.5
		for i in xrange( start_scale, max_pyra + 1 ) :
			if type(pos_list[i-start_scale]) == list :
				pos_list[i-start_scale] = apply_nms( pos_list[i-start_scale], thre_cal1, 'inscale' )
			else :
				pos_list[i-start_scale] = apply_nms( pos_list[i-start_scale].tolist(), thre_cal1, 'inscale' )

	# Detection 24 net
	if len(network_list) >= 3 :
		for i in xrange(start_scale, max_pyra + 1 ) :
			scale_up = i
			pos_list[i-start_scale] = apply_det( network_list, pos_list[i-start_scale], pyra[i-scale_up], threshold2, '24net', scale_factor, scale_up)

	if len(network_list) >= 4 :
		# Calibration 24 net
		for i in xrange( start_scale, max_pyra + 1 ) :
			scale_up = i
			pos_list[i-start_scale] = apply_calib( network_list[3], pos_list[i-start_scale], pyra[i-scale_up], '24net', scale_factor, scale_up)

		# In-Scale NMS
		thre_cal2 = 0.5
		for i in xrange( start_scale, max_pyra + 1 ) :
			pos_list[i-start_scale] = apply_nms( pos_list[i-start_scale], thre_cal2, 'inscale' )

	# Detection 24 net
	if len(network_list) >= 5 :
		for i in xrange( start_scale, max_pyra + 1 ) :
			scale_up = i
			pos_list[i-start_scale] = apply_det( network_list, pos_list[i-start_scale], pyra[i-scale_up], threshold3, '48net', scale_factor, scale_up)

	# # display_patch( pyra[0], np.divide(tmp_annot_list[0], resize_ratio) )
	# test_scale = 5
	# for i in xrange(10) :
	# 	display_patch( pyra[test_scale], pos_list[test_scale][i])

	# ADJUST SCALE and MERGE
	for i in xrange( start_scale, max_pyra + 1 ) :
		pos_list[i-start_scale] = np.array(pos_list[i-start_scale])
		pos_list[i-start_scale][:,0:4] = np.multiply( pos_list[i-start_scale][:,0:4], scale_factor**i )

	final_list = []
	for i in xrange( start_scale, max_pyra + 1 ) :
		if pos_list[i-start_scale][0][2] > 0 :		# means there's face in this pyramid
			if len(final_list) == 0 :
				final_list = pos_list[i-start_scale]
			else :
				final_list = np.concatenate( [ final_list, pos_list[i-start_scale] ], axis = 0 )

	# Global NMS
	if len(network_list) >= 6 :
		if not len(final_list)==0 :
			# thre_cal3 = 0.2
			thre_cal3 = 0.7
			final_list = apply_nms( final_list.tolist(), thre_cal3, 'global' )

	# Calibration 48 net
	if len(network_list) >= 6 :
		if not len(final_list)==0 :
			scale_up = 0
			final_list = apply_calib( network_list[5], final_list, pyra[0], '48net', scale_factor, scale_up)

	if not len(final_list)==0 :
		final_list = np.array( final_list )
		final_list[:,0:4] = np.multiply( final_list[:,0:4], resize_ratio )

	if func_mode == 'table' :
		true_pos = 0
		for i in xrange(len(final_list) ) :
			true_pos = true_pos + check_iou(final_list[i], tmp_annot_list, 0.5, 0)
		
		return true_pos, len(final_list)
	elif func_mode == 'annot' :
		return final_list

def detect_pos (network_list, tmp_img, threshold1, threshold2, dataset_name, resize_ratio ) :

	if dataset_name == 'aflw' :
		scale_factor = 1.414
	elif dataset_name == 'fddb' :
		scale_factor = 1.18

	start_scale = 0
	pyra = list( pyramid_gaussian(tmp_img, downscale=scale_factor ) )
	max_pyra = remove_small_pyra( pyra )

	if len(pyra[0].shape)!=3 :
		tmp_img = tmp_img.convert('RGB')
		pyra = list( pyramid_gaussian(tmp_img, downscale=scale_factor ) )
		max_pyra = remove_small_pyra( pyra )
	
	pos_list = []
	# Detection 12 net
	for i in xrange( max_pyra +1 ) :
		pos_list.append( scan_12net(network_list[0], pyra[i], threshold1, i, scale_factor, 0) )

	if len(network_list) >= 2 :
		# Calibration 12 net
		for i in xrange( max_pyra +1 ) :
			scale_up = 0
			pos_list[i] = apply_calib( network_list[1], pos_list[i], pyra[i-scale_up], '12net', scale_factor, scale_up )

		# In-Scale NMS
		# thre_cal1 = 0.5
		thre_cal1 = 0.7
		for i in xrange( start_scale, max_pyra + 1 ) :
			if type(pos_list[i-start_scale]) == list :
				pos_list[i-start_scale] = apply_nms( pos_list[i-start_scale], thre_cal1, 'inscale' )
			else :
				pos_list[i-start_scale] = apply_nms( pos_list[i-start_scale].tolist(), thre_cal1, 'inscale' )

	if len(network_list) >= 3 :
		# Detection 24 net
		for i in xrange( max_pyra + 1 ) :
			scale_up = i
			pos_list[i] = apply_det( network_list, pos_list[i], pyra[i-scale_up], threshold2, '24net', scale_factor, scale_up)

	if len(network_list) >= 4 :
		prev_count = 0
		for i in xrange( max_pyra + 1 ) :
			prev_count = prev_count + len(pos_list[i])
		# Calibration 24 net
		for i in xrange( max_pyra + 1 ) :
			scale_up = i
			pos_list[i] = apply_calib( network_list[3], pos_list[i], pyra[i-scale_up], '24net', scale_factor, scale_up)

		# In-Scale NMS
		thre_cal2 = 0.5
		for i in xrange( max_pyra + 1 ) :
			pos_list[i] = apply_nms( pos_list[i], thre_cal2, 'inscale' )


	# Unify spatial scale
	for i in xrange( max_pyra +1 ) :
		pos_list[i] = np.array(pos_list[i])
		pos_list[i][:,0:4] = np.multiply( pos_list[i][:,0:4], resize_ratio * scale_factor**i )

	final_list = []
	for i in xrange( max_pyra +1 ) :
		if pos_list[i][0][2] > 0 :		# means there's no face in this pyramid
			if len(final_list) == 0 :
				final_list = pos_list[i]
			else :
				final_list = np.concatenate( [ final_list, pos_list[i] ], axis = 0 )

	return final_list

def scan_12net (N12, tmp_pyra, threshold, scale, scale_factor, ns_type) :
	tmp_pyra = tmp_pyra
	img_shape = np.shape(tmp_pyra)

	if scale_factor < 1.2 :
		ref_scale1 =9
		ref_scale2 =14
	else :
		ref_scale1 = 5
		ref_scale2 = 6

	if scale >= ref_scale2 :
		stride = 1
	elif scale >= ref_scale1 :
		stride = 2
	else :
		stride = 4
	# stride = 4

	# if ns_type == 1 :
	# 	stride = 12

	logit_len_x = int( (img_shape[1]-12)/stride ) + 1
	logit_len_y = int( (img_shape[0]-12)/stride ) + 1

	size_window = logit_len_x * logit_len_y 
	patch_list = np.zeros( [size_window, 12*12*3]).astype(np.float32)
	cnt = 0

	for j in xrange(0, img_shape[0] - 12 + 1, stride ) :
		for i in xrange(0, img_shape[1] - 12 + 1, stride) :
			try :
				tmp_pyra[j, i]
				tmp_pyra[j+11, i+11]
			except IndexError :
				print np.shape(tmp_pyra)
				print j, i
				print logit_len_x, logit_len_y
				sys.exit("IndexError")
			patch = np.array( tmp_pyra[j:j+12, i:i+12] )
			patch = np.reshape(patch, [1, 12*12*3] )
			patch_list[cnt] = patch
			cnt = cnt + 1

	pos_list = np.zeros([logit_len_y*logit_len_x, 5]).astype(np.float16)
	if not mode_table == 'sliding' :
		logit = N12.y_conv.eval( feed_dict = { N12.x:patch_list} )
		logit = np.reshape( logit, [logit_len_y, logit_len_x] )
	cnt = 0

	for j in xrange(logit_len_y) :
		for i in xrange(logit_len_x) :
			if mode_table == 'sliding' :
				left	= i*stride		# stride 4
				btm		= j*stride
				width	= 12
				height	= 12
				tmp_annot = [left, btm, width, height, 0 ] 
				pos_list[cnt] = tmp_annot
				cnt = cnt + 1
				
			else :
				if logit[j,i] >= threshold :
					left	= i*stride		# stride 4
					btm		= j*stride
					width	= 12
					height	= 12
					tmp_annot = [left, btm, width, height, logit[j,i] ] 
					pos_list[cnt] = tmp_annot
					cnt = cnt + 1
					# pos_list.append( tmp_annot )
					if int(left) == 317 :
						print "somthing wrong damn.."
						print left, btm
						print logit_len_y, logit_len_x
						print j, i

		
	if cnt == 0 :
		pos_list = [ [-1,-1,-1,-1, -1] ]	# means theres no face
	else :
		pos_list = pos_list[0:cnt]

	return pos_list

def apply_calib (CAL, pos_list, tmp_pyra, netname, scale_factor, scale_up ) :
	tmp_list = []
	# If threre's no detected face in this pyramid, bypass with negative notation
	# print np.shape(pos_list)
	# print pos_list[0]
	# print np.shape(pos_list[0])
	if pos_list[0][2] < 0 :
		return pos_list

	patch_list = []
	tmp_pyra_shape = np.shape(tmp_pyra)
	for k in xrange(len(pos_list) ) :
		adjust_size = scale_factor ** scale_up
		left	= int( pos_list[k][0] * adjust_size )
		btm		= int( pos_list[k][1] * adjust_size )
		width	= int( pos_list[k][2] * adjust_size )
		height	= int( pos_list[k][3] * adjust_size )

		left_b = 0
		btm_b = 0
		right_b = 0
		top_b = 0
		if left < 0 :
			left_b = -left
			left = 0
		if btm < 0 :
			btm_b = -btm
			btm = 0
		if left+width > tmp_pyra_shape[1] :
			right_b = left+width - tmp_pyra_shape[1]
		if btm+height > tmp_pyra_shape[0] :
			top_b = btm+height - tmp_pyra_shape[0]

		patch = tmp_pyra[btm:btm+height, left:left+width ]
		crop_size = np.shape(patch)
		patch = np.concatenate( [np.zeros([crop_size[0], left_b,3 ]), patch, np.zeros([crop_size[0], right_b,3]) ] , axis = 1 )
		crop_size = np.shape(patch)
		patch = np.concatenate( [np.zeros([btm_b, crop_size[1],3  ]), patch, np.zeros([top_b, crop_size[1],3  ]) ] , axis = 0 )

		if netname == '12net' :
			patch = scipy.misc.imresize( patch, [ 12, 12 ],  interp = 'bilinear')
			patch = np.reshape(patch, [1, 12*12*3] )
		elif netname == '24net' :
			patch = scipy.misc.imresize( patch, [ 24, 24 ],  interp = 'bilinear')
			patch = np.reshape(patch, [1, 24*24*3] )
		elif netname == '48net' :
			patch = scipy.misc.imresize( patch, [ 48, 48 ],  interp = 'bilinear')
			patch = np.reshape(patch, [1, 48*48*3] )

		if len(patch_list)==0 :
			patch_list = patch
		else :
			patch_list = np.concatenate( [patch_list, patch], axis = 0 )

	logits = CAL.y_conv.eval( feed_dict={CAL.x: patch_list} )

	for k in xrange(len(pos_list) ) :
		trans_index = np.argmax( logits[k] )
		s_index, x_index, y_index = mapping_calib_para ( trans_index )
		s, x, y = decoding_calib ( s_index, x_index, y_index )
		tmp_annot = np.concatenate( [transform_calib( pos_list[k], s, x, y ), [pos_list[k][4]] ], axis = 0 )
		# tmp_annot = np.concatenate( [convert_invcalib( pos_list[k], s, x, y ), [pos_list[k][4]] ], axis = 0 )
		tmp_list.append( tmp_annot )

	return tmp_list

def apply_det (network_list, pos_list, tmp_pyra, threshold, netname, scale_factor, scale_up ) :
	N12 = network_list[0]
	N24 = network_list[2]
	if netname == '48net' :
		N48 = network_list[4]

	tmp_list = []
	# If threre's no detected face in this pyramid, bypass with negative notation
	if pos_list[0][2] < 0 :
		return pos_list

	patch_list = []
	patch_list_12net = []
	patch_list_24net = []
	tmp_pyra_shape = np.shape(tmp_pyra)
	for k in xrange(len(pos_list) ) :
		adjust_size = scale_factor ** scale_up
		left	= int( pos_list[k][0] * adjust_size )
		btm		= int( pos_list[k][1] * adjust_size )
		width	= int( pos_list[k][2] * adjust_size )
		height	= int( pos_list[k][3] * adjust_size )

		left_b = 0
		btm_b = 0
		right_b = 0
		top_b = 0
		if left < 0 :
			left_b = -left
			left = 0
		if btm < 0 :
			btm_b = -btm
			btm = 0
		if left+width > tmp_pyra_shape[1] :
			right_b = left+width - tmp_pyra_shape[1]
		if btm+height > tmp_pyra_shape[0] :
			top_b = btm+height - tmp_pyra_shape[0]

		pyra_crop = tmp_pyra[btm:btm+height, left:left+width ]
		crop_size = np.shape(pyra_crop)
		pyra_crop = np.concatenate( [np.zeros([crop_size[0], left_b,3 ]), pyra_crop, np.zeros([crop_size[0], right_b,3]) ] , axis = 1 )
		crop_size = np.shape(pyra_crop)
		pyra_crop = np.concatenate( [np.zeros([btm_b, crop_size[1],3  ]), pyra_crop, np.zeros([top_b, crop_size[1],3  ]) ] , axis = 0 )

		if netname == '24net' :
			patch = scipy.misc.imresize( pyra_crop, [ 24, 24 ],  interp = 'bilinear')
			patch = np.reshape(patch, [1, 24*24*3] )
		elif netname == '48net' :
			patch = scipy.misc.imresize( pyra_crop, [ 48, 48 ],  interp = 'bilinear')
			patch = np.reshape(patch, [1, 48*48*3] )
			patch_24net = scipy.misc.imresize( pyra_crop, [ 24, 24 ],  interp = 'bilinear')
			patch_24net = np.reshape(patch_24net, [1, 24*24*3] )
		patch_12net = scipy.misc.imresize( pyra_crop, [ 12, 12 ],  interp = 'bilinear')
		patch_12net = np.reshape(patch_12net, [1, 12*12*3] )

		if len(patch_list)==0 :
			patch_list = patch
			patch_list_12net = patch_12net
			if netname == '48net' :
				patch_list_24net = patch_24net
		else :
			patch_list = np.concatenate( [patch_list, patch], axis = 0 )
			patch_list_12net = np.concatenate( [patch_list_12net, patch_12net], axis = 0 )
			if netname == '48net' :
				patch_list_24net = np.concatenate( [patch_list_24net, patch_24net], axis = 0 )

	if netname == '24net' :
		ccd_12net = N12.h_fc1.eval( feed_dict = {N12.x: patch_list_12net} )
		logits = N24.y_conv.eval( feed_dict={N24.x: patch_list, N24.in_12net : ccd_12net} )
		logits = np.reshape(logits, [len(pos_list)] )
	elif netname == '48net' :
		ccd_12net = N12.h_fc1.eval( feed_dict = {N12.x: patch_list_12net} )
		ccd_24net_1 = N24.h_fc1.eval( feed_dict={ N24.x:patch_list_24net, N24.in_12net:ccd_12net } )
		ccd_24net = np.concatenate( [ccd_12net, ccd_24net_1], axis = 3 )
		logits = N48.y_conv.eval( feed_dict={N48.x: patch_list, N48.in_24net: ccd_24net } )
		logits = np.reshape(logits, [len(pos_list)] )

	for k in xrange(len(pos_list) ) :
		if( logits[k] > 1 ) :
			sys.exit("logits value error!!")
		elif( logits[k] >= threshold ) :
			tmp_annot = pos_list[k]
			tmp_annot[4] = logits[k]
			tmp_list.append( tmp_annot )

	if len(tmp_list)==0 :
		tmp_list = [ [-1,-1,-1,-1, -1] ]	# means theres no face

	return tmp_list

def apply_nms ( pos_list, thre, cond_type ) :
	final_list = []

	# print pos_list[0]
	if pos_list[0][2] < 0 :
		return pos_list
	
	while len(pos_list)!=0 :
		temp = pos_list.pop()
		flag = 0

		temp_list = copy.deepcopy( pos_list )

		for i in range( len(temp_list) ) :
			temp2 = temp_list.pop()
			# print "current_len", i, len(pos_list), len(temp_list)
			cond_near = check_neighbor(temp, temp2, thre, cond_type)

			if cond_near :
				cond_w = temp[4] >= temp2[4]
				if cond_w :
					# print "temp type : ", type(temp), len(temp), temp
					# print "temp2 type : ", type(temp2), len(temp2), temp2
					# print "pos_list type : ", type(pos_list), len(pos_list)
					removearray(pos_list, temp2)
					pass
				else :
					flag = 1
					break

		if flag==0 :	# it means temp has biggest logit among its neighbors
			final_list.insert(0, temp)

	if len(final_list) == 0 :
		sys.exit("apply nms error!!")
	return final_list

def removearray(array_list, array) :
	for i in xrange(len(array_list)) :
		if(np.array_equal(array_list[i], array) ) :
			array_list.__delitem__(i)
			return 1

	return 0 # there's no matched item

def decoding_annot ( tmp_line ) :
	cond_eof = ( tmp_line == '\n' ) | ( tmp_line == '' )
	cond_numface = ( len(tmp_line.split()) == 1 ) & ( len(tmp_line.split('/')) == 1  )
	
	if cond_numface != 1 :
		cond_newimg = len(tmp_line.split('/'))>=2
	else :
		cond_newimg = 0
	
	if cond_eof + cond_numface + cond_newimg == 0 :
		cond_annot = 1
	else :
		cond_annot = 0

	return cond_newimg, cond_numface, cond_annot, cond_eof

def display_patch( img, annot ) :
	left	= int( annot[0] )
	btm		= int( annot[1] )
	width	= int( annot[2] )
	height	= int( annot[3] )
	# img.crop( (left, btm, left+width, btm+height) ).show()
	tmp_array = np.multiply(img[btm:btm+height, left:left+width], 255.).astype(np.uint8)
	tmp_img = Image.fromarray(tmp_array)
	tmp_img.show()
	return 1

def save_patch( img, annot, path, net_name ) :
	left	= int( annot[0] )
	btm		= int( annot[1] )
	width	= int( annot[2] )
	height	= int( annot[3] )
	tmp_img = img.crop( (left, btm, left+width, btm+height) )
	if net_name == '12net' :
		tmp_img.resize( (12, 12), Image.BILINEAR ).save(path)
	elif net_name == '24net' :
		tmp_img.resize( (24, 24), Image.BILINEAR ).save(path)
	elif net_name == '48net' :
		tmp_img.resize( (48, 48), Image.BILINEAR ).save(path)
	return 1

def save_annot( annot_list ) :
	print "debug annot list!"
	debug_file = open("debug_annot.txt", 'w')
	for i in xrange(len(annot_list) ) :
		debug_file.write("%s\n"% annot_list[i])

# def remove_small_pyra ( pyra ) :
# 	for i in range(len(pyra) ):
# 		if min( pyra[i].shape[0], pyra[i].shape[1] ) < 12 :
# 			del pyra[i:]
# 			break
# 	return pyra

def remove_small_pyra ( pyra ) :
	for i in range(len(pyra) ):
		if min( pyra[i].shape[0], pyra[i].shape[1] ) < 12 :
			return i-1
			# del pyra[i:]
			# break
	# return pyra

def check_iou (tmp_patch, annot_list, thre, ns_type ) :
	sq1_left	= int( tmp_patch[0] )
	sq1_btm		= int( tmp_patch[1] )
	sq1_width	= int( tmp_patch[2] )
	sq1_height	= int( tmp_patch[3] )

	sq1_right = sq1_left + sq1_width
	sq1_top   = sq1_btm  + sq1_height * 1.1
	sq1_btm   = sq1_btm  - sq1_height * 0.1

	for i in xrange(len(annot_list) ) :
		if (annot_list[i][4] == 0) | ns_type :
			square2 = annot_list[i]
			sq2_left	= int( square2[0] )
			sq2_btm		= int( square2[1] )
			sq2_width	= int( square2[2] )
			sq2_height	= int( square2[3] )

			sq2_right = sq2_left + sq2_width
			sq2_top   = sq2_btm  + sq2_height

			# IOU : Intersection Over Union
			if min(sq1_right,sq2_right) < max(sq1_left,sq2_left) :
				A_inter = 0
			elif min(sq1_top,sq2_top) < max(sq1_btm,sq2_btm) :
				A_inter = 0
			else :
				A_inter = ( min(sq1_right,sq2_right) -max(sq1_left,sq2_left) ) * ( min(sq1_top,sq2_top) -max(sq1_btm,sq2_btm) )

			sq1_size = (sq1_right -sq1_left) * (sq1_top -sq1_btm)
			sq2_size = (sq2_right -sq2_left) * (sq2_top -sq2_btm)
			A_union = sq1_size + sq2_size - A_inter
			IOU = float(A_inter) / float(A_union)

			# if IOU >= 0.5 :
			if IOU >= thre :
				annot_list[i][4] = 1
				return 1		# new face detected
			
			if ns_type == 1 :
				# Belong & Include
				cond_inc = 0
				cond_bel = 0
				if not sq2_size == 0 :
					cond_inc = ( float(A_inter) / float(sq2_size) ) > 0.8
				# if not sq1_size == 0 :
				# 	cond_bel = ( float(A_inter) / float(sq1_size) ) > 0.8
				# cond_inc = 0

				if ( cond_inc | cond_bel ) : 
					return 1

	else :
		return 0	# this square is not a face

def check_neighbor (square1, square2, thre, cond_type) :
	sq1_left = int( square1[0] )
	sq1_btm  = int( square1[1] )
	sq1_width  = int( square1[2] )
	sq1_height = int( square1[3] )

	sq2_left = int( square2[0] )
	sq2_btm  = int( square2[1] )
	sq2_width  = int( square2[2] )
	sq2_height = int( square2[3] )

	# IOU : Intersection Over Union
	sq1_right = sq1_left + sq1_width
	sq1_top   = sq1_btm  + sq1_height
	sq2_right = sq2_left + sq2_width
	sq2_top   = sq2_btm  + sq2_height

	if min(sq1_right,sq2_right) < max(sq1_left,sq2_left) :
		A_inter = 0
	elif min(sq1_top,sq2_top) < max(sq1_btm,sq2_btm) :
		A_inter = 0
	else :
		A_inter = ( min(sq1_right,sq2_right) -max(sq1_left,sq2_left) ) * ( min(sq1_top,sq2_top) -max(sq1_btm,sq2_btm) )

	sq1_size = (sq1_right -sq1_left) * (sq1_top -sq1_btm)
	sq2_size = (sq2_right -sq2_left) * (sq2_top -sq2_btm)
	A_union = sq1_size + sq2_size - A_inter
	IOU = float(A_inter) / float(A_union)

	cond_iou = IOU >= thre

	# Belong & Include
	# cond_inc = ( float(A_inter) / float(sq2_size) ) > 0.8
	# cond_bel = ( float(A_inter) / float(sq1_size) ) > 0.8
	cond_inc = ( float(A_inter) / float(sq2_size) ) > 0.9
	cond_bel = ( float(A_inter) / float(sq1_size) ) > 0.9

	if cond_iou :	  # IOU
		return 1
	elif (cond_type == 'global') & ( cond_inc | cond_bel ) : 
		return 1
	else :
		return 0



