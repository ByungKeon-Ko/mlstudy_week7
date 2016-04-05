# 1. Purpose
#	   find and generate positive sample of which size is 32x32
#

import Image
import numpy as np

# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/week7"
aflw_path = "%s/../aflw/aflw" %base_path

# net_name = '12net'
# net_name = '24net'
net_name = '48net'

if net_name == '12net' :
	FACESIZE = 12
	ps_path = "PS_det12"
elif net_name == '24net' :
	FACESIZE = 24
	ps_path = "PS_det24"
elif net_name == '48net' :
	FACESIZE = 48
	ps_path = "PS_det48"

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

def gen_ps(temp_line, total_ps) :
	global org_img

	# print "gen_ps function!"
	# temp_line = temp_line.split(',')
	left_x =  int( temp_line[0] )
	btm_y  =  int( temp_line[1] )
	width  =  int( temp_line[2] )
	height =  int( temp_line[3] )

	center_x	= left_x + width/2
	center_y	= btm_y + height/2
	right_x		= left_x + width
	top_y		= btm_y  + height

	if min(width, height) < 10 :
		return 0

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

	# pos_img = pos_img.resize( (FACESIZE,FACESIZE), Image.BICUBIC )
	pos_img = pos_img.resize( (FACESIZE,FACESIZE), Image.BILINEAR )
	# pos_img.show()
	pos_img.save( "%s/ps-%d.jpg" %(ps_path, total_ps) )
	if total_ps %4000 == 0 :
		print "generate %dth-positive sample!" %total_ps
	return 1

# ----------------------------------------- 

total_ps = 0
rect_file = open("%s/annotation/annot_ksmoon" %(aflw_path), 'r')
line_matrix = rect_file.readlines()

print "PreparePS.py start!!, net_name : ", net_name

for i in range(0, TotalNum) :

	temp_line = line_matrix[i].rstrip()
	temp_path = temp_line.split(',')[0]
	temp_annot = temp_line.split(',')[1:5]
	
	if load_orgimg(temp_path) == 0 :
		continue
	if gen_ps(temp_annot, total_ps) == 1 :
		total_ps = total_ps + 1

