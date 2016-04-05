# 1. Purpose
#		save pyramid images so they can be used for generating negative samples in train.py
#
#	   load images in rect-#.txt list, generating pyramid images and counting # of MinFacexMinFace patches
#	   and save them in "patches" folder
#	   Also, record each original image name and location of patches in pyramid_index.txt
#	   so that we can use pyramid_index.txt file for generating negative sample ( negative sample will be rejected according to IOU value on train.py )
#
# 2. output files
#	   pyramid-#.jpg : pyramid images
#	   pyramid_index.txt
#		   <# of patch> <scale> <left_x> <top_y> <width> <height>

import Image
import numpy as np
import math

from skimage.transform import pyramid_gaussian

# ------ Parameters ---------------------- 
base_path		= "/home/bkko/ml_study/week7"
ns_path			= "%s/pyramids" % base_path
indexfile_path	= "%s/pyramid_index.txt"   %ns_path
annot_path		= "%s/../fddb/FDDB-folds"	  %(base_path)

fold_num = 8
MinFace = 48

# ------ functions ------------------------
patchNum = 0
i = 0
rect_loc = 0
annotfile_num = 0
orgimg_path = 0
total_patch = 0
total_pyramid = 0

def write_contents () :
	global cnt_patch
	global patchNum
	global rect_loc
	global pyramids

	# print "write_contents!"
	for i in range( len(pyramids) ) :
		index_file.write("%d " %patchNum[i] )	# num of patches
		index_file.write("%d " %i)				# scale
		index_file.write("%s " %rect_loc)		# rectangle location
		index_file.write("%s " %annotfile_num)		# annotation file #
		index_file.write("%s \n" %orgimg_path)	# original image path
	cnt_patch = cnt_patch + len(pyramids)

def save_pyramid () :
	global temp_line
	global pyramids
	global patchNum
	global total_patch
	global total_pyramid

	org_img = Image.open("%s/../fddb/%s.jpg" %(base_path, temp_line), 'r' )
	
	org_img_name = "%s " %(temp_line)		# original image name
	
	pyramids = list( pyramid_gaussian(org_img, downscale=math.sqrt(2) ) )
	for i in range(len(pyramids) ):
		if min( pyramids[i].shape[0], pyramids[i].shape[1] ) < MinFace :
			del pyramids[i:]
			break
	
	for i in range( len (pyramids) ) :
		row = pyramids[i].shape[0]
		col = pyramids[i].shape[1]
		im_matrix = np.zeros([row, col, 3]).astype('uint8')
	
		for k in range(row):
			for j in range(col):
				im_matrix[k,j] = pyramids[i][k,j] * 255
	
		new_img = Image.fromarray(im_matrix)
		new_img.save("%s/pyramid-%s.jpg" %(ns_path, i+total_pyramid) )
		# new_img.show()
	
		patchNum[i] = (row-MinFace+1) * (col-MinFace+1)				  # the number of patches
	total_pyramid = total_pyramid + len(pyramids)
	total_patch = total_patch + sum(patchNum)

# ----------------------------------------- 
index_file   = open(indexfile_path, 'w')

for i in range(1,fold_num+1) :
	annotfile_num = i
	rect_file_path  = "%s/rect-%s.txt"	  %(annot_path, i)
	rect_file  = open(rect_file_path,  'r')

	cnt_patch = 0

	patchNum = np.zeros( [10] ).astype('uint32')

	rect_loc = 0
	cnt_img = 0
	while 1 :
		if cnt_img%10 == 0 :
			print "current img : ", total_pyramid, cnt_img, total_patch
		temp_line = rect_file.readline().rstrip()

		cond_eof = temp_line == ''
		cond_numface = len(temp_line) <= 2
		if cond_numface != 1 :
			cond_newimg = (temp_line[0:3] == '200') & (temp_line[3]!=' ') & (temp_line[3]!='.')
		else :
			cond_newimg = 0

		if cond_eof :	   # end of file
			write_contents()
			print "1 set done!!", i
			break

		elif cond_numface : # the number of face in the image
			pass

		elif cond_newimg :  # new image name
			if ( cnt_img != 0 ) :   # at the first case
				write_contents()

			cnt_img = cnt_img + 1
			orgimg_path = temp_line
			save_pyramid()

		else :			  # face location
			rect_loc = temp_line

index_file.write ("totalsum:%d" %total_patch )
print "index writing done!!"


