
import Image
import numpy as np
import math
import random

from skimage.transform import pyramid_gaussian

# ------ Parameters ---------------------- 
base_path		= "/home/bkko/ml_study/week7"
ns_path			= "%s/NS_det12" % base_path
coco_path		= "%s/../coco" %base_path

MinFace		= 20
TotalNum	= 5764

# ------ functions ------------------------
def create_ns (tmp_imgpath, cnt_ns ) :
	global pyramids

	tmp_img = Image.open("%s/%s" %(coco_path, tmp_imgpath), 'r' )
	pyramids = list( pyramid_gaussian( tmp_img, downscale=math.sqrt(2) ) )

	for i in range ( len(pyramids) ):
		if min( pyramids[i].shape[0], pyramids[i].shape[1] ) < MinFace :
			del pyramids[i:]
			break

	# for j in range(4) :
	for j in range(36) :
		# creating random index
		img_index = random.randint(0, len(pyramids)-1 )
		tmp_patch_num = ( pyramids[img_index].shape[0] - 12 + 1) * ( pyramids[img_index].shape[1] - 12 + 1)
		rand_index = random.randint(0, tmp_patch_num)

		# x, y position decoding
		row_max = pyramids[img_index].shape[0]
		col_max = pyramids[img_index].shape[1]
		row = 0
		col = rand_index
		
		while ( col >= col_max - 12 +1 ) :
			row = row + 1
			col = col - (col_max-12+1)

		flag = 0
		# Rejecting Black and White image
		tmp_ns = pyramids[img_index][row:row+12, col:col+12]
		if not len(tmp_ns.shape)==3 :
			print " Gray Image. Skip "
			return 0

		# Rejecting Positive Samples
		scale_factor = math.sqrt(2)**img_index

		tmp_ns = pyramids[img_index][row:row+12, col:col+12]
		tmp_ns = Image.fromarray((tmp_ns*255.0).astype(np.uint8) )
		# tmp_ns = tmp_ns.resize( (12,12), Image.BICUBIC )
		tmp_ns = tmp_ns.resize( (12,12), Image.BILINEAR )
		tmp_ns.save("%s/ns-%s.jpg" %(ns_path, cnt_ns+j) )

	return 1

# ----------------------------------------- 
cnt_ns			= 0
tmp_imgpath		= 0
rect_file_path	= "%s/file_list"	  %(coco_path)
rect_file		= open(rect_file_path,  'r').readlines()

for i in range(1,TotalNum+1) :
	tmp_imgpath = rect_file[i-1].rstrip()

	if create_ns(tmp_imgpath, cnt_ns )==1 :
		# cnt_ns = cnt_ns + 4
		cnt_ns = cnt_ns + 36
	else :
		print "Fail to create NS!! @ %d" %i

	if i%100 == 0 :
		print "current step : ", i

print "index writing done!!"


