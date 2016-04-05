# 1. Purpose
#	   find and generate positive sample of which size is 32x32
#

import Image
import numpy as np
import random
from CalibLib import mapping_calib_para
from CalibLib import decoding_calib
from CalibLib import convert_invcalib
from CalibLib import transform_calib 
import time

# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/week7"
aflw_path = "%s/../aflw/aflw" %base_path
# net_name = '12net'
# net_name = '24net'
net_name = '48net'
# ps_path = "PS_calib"

if net_name == '12net' :
	ps_path = "PS_cal12"
	FACESIZE = 12
elif net_name == '24net' :
	ps_path = "PS_cal24"
	FACESIZE = 24
elif net_name == '48net' :
	ps_path = "PS_cal48"
	FACESIZE = 48

TotalNum = 24384

# ------ functions ------------------------

def load_orgimg(temp_line) :
	global org_img
	global org_file
	global base_path
	global org_width, org_height

	# print "load_orgimg function!"
	org_file = Image.open("%s/data/flickr/%s" %(aflw_path, temp_line), 'r' )
	try:
		org_img = org_file.load()
	except IOError:
		print "IOError happens!! because of Image load"
		return 0
	org_width, org_height = org_file.size

	return 1

new_square = []


def gen_ps(temp_annot, total_ps, calib_annot_file) :
	global org_img

	# param_index = random.randint(0, 45)
	for param_index in xrange(45) :
		index_s, index_x, index_y = mapping_calib_para(param_index)
		org_s, org_x, org_y = decoding_calib( index_s, index_x, index_y )
		new_square = convert_invcalib( temp_annot, org_s, org_x, org_y )
	
		left_x	= new_square[0]
		btm_y	= new_square[1]
		width	= new_square[2]
		height	= new_square[3]
	
		center_x	= left_x + width/2
		center_y	= btm_y + height/2
		# right_x		= left_x + width
		# top_y		= btm_y  + height
	
		# if min(width, height) < 10 :
		# 	break
	
		ps_len = max(width, height)
		if ps_len > min(org_width, org_height) :
			ps_len = min(org_width, org_height)
	
		ps_x = center_x -ps_len/2
		if ps_x < 0 :
			ps_len = center_x - left_x -1
			ps_x = center_x -ps_len/2
	
		ps_y = center_y -ps_len/2
		if ps_y < 0 :
			ps_len = center_y - btm_y -1
			ps_y = center_y -ps_len/2
			ps_x = center_x -ps_len/2
	
		# Overflow & Underflow index preventing
		if (ps_x + ps_len) >= org_width :
			ps_x = org_width - ps_len - 1
		if ps_y < 0 :
			ps_y = 0
		if (ps_y + ps_len) >= org_height :
			ps_y = org_height - ps_len - 1
		if ps_y < 0 :
			ps_y = 0
		if ps_x < 0 :
			ps_x = 0
	
		tmp_imfile = org_file
		org_load = tmp_imfile.load()
	
		if np.shape(org_load[0,0]) == () :
			tmp_imfile = tmp_imfile.convert('RGB')
			org_load = tmp_imfile.load()
			# print " Gray Image Convert "
		elif not len(org_load[0,0]) == 3 :
			tmp_imfile = tmp_imfile.convert('RGB')
			org_load = tmp_imfile.load()
			# print " Gray Image Convert "
	
		pos_img = tmp_imfile.crop( (int(ps_x), int(ps_y ), int(ps_x + ps_len), int(ps_y+ps_len) ) )
	
		pos_img = pos_img.resize( (FACESIZE,FACESIZE), Image.BILINEAR )
		pos_img.save( "%s/ps-%d.jpg" %(ps_path, total_ps+param_index) )
		calib_annot_file.write("ps-%d.jpg,%d,%d,%d\n" %(total_ps+param_index, index_s, index_x, index_y ) )
		
	return 1

# ----------------------------------------- 

total_ps = 0
# total_ps = 8160
rect_file = open("%s/annotation/annot_ksmoon" %(aflw_path), 'r')
# calib_annot_file = open("%s/calib_annot.txt" %(ps_path), 'w' )
calib_annot_file = open("%s/calib_annot.txt" %(ps_path), 'a' )
# calib_annot_file = open("%s/calib_annot_2ndhalf.txt" %(ps_path), 'a' )
temp_line = rect_file.readlines()

print "PrepareCalib.py start!!, net_name : ", net_name

# start_time = time.time()
for i in range(0, TotalNum) :
# for i in range(8160, TotalNum) :

	tmp	= temp_line[i].rstrip()
	temp_path = tmp.split(',')[0]
	temp_annot = tmp.split(',')[1:5]
	
	if load_orgimg(temp_path) == 0 :
		continue
	# print "load time : ", time.time() - start_time
	# start_time = time.time()
	if gen_ps(temp_annot, total_ps, calib_annot_file) == 1 :
		total_ps = total_ps + 45


	if i%2000 == 0 :
		print "generate %dth-positive sample!" %i
		# print "gen_ps : ", time.time() - start_time
		# start_time = time.time()

