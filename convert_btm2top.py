import image_search
import numpy as np
import random
import Image

# btm_file_path  = "/home/bkko/ml_study/fddb/FDDB-folds/rectbtm-1.txt"
# btm_file_path  = "/home/bkko/ml_study/week7/ofile_12net/o_annot.txt"
btm_file_path  = "/home/bkko/ml_study/week7/ofile_12net/o_annot_10.txt"
top_file_path  = "/home/bkko/ml_study/week7/Figure9/fold-10-out.txt"
fddb_path		= "/home/bkko/ml_study/fddb"
btm_file = open(btm_file_path, 'r' ).readlines()
top_file = open(top_file_path, 'w' )

btm_file_index = 0
tmp_annot = np.zeros([5]).astype(np.float)

while 1 :
	if btm_file_index == len(btm_file) :
		break
	tmp_line = btm_file[btm_file_index].rstrip()
	btm_file_index = btm_file_index + 1
	cond_newimg, cond_numface, cond_annot, cond_eof = image_search.decoding_annot(tmp_line ) 

	if cond_eof :
		break

	if cond_newimg :
		tmp_img_path = tmp_line.rstrip()
		tmp_img = Image.open("%s/%s.jpg"%(fddb_path, tmp_img_path) )
		img_size = tmp_img.size
	elif cond_numface :
		tmp_numface = int( tmp_line.rstrip() )
		tmp_annot_list = np.zeros([tmp_numface,5]).astype(np.float32)
		annot_cnt = 0
	elif cond_annot :
		tmp_annot[0:4] = np.array( tmp_line.split()[0:4] ).astype(np.int16)
		# logit_value = random.randint(0,255)/255.
		logit_value = tmp_line.split()[4]

		left	= int(tmp_annot[0] )
		btm		= int(tmp_annot[1] )
		width	= int(tmp_annot[2] )
		height	= int(tmp_annot[3] )

		# if left < 0 :
		# 	width	= width + left
		# 	left	= 0
		# if btm < 0 :
		# 	height	= height + btm
		# 	btm		= 0

		# right	= left + width
		# if right >= img_size[0] :
		# 	right	= img_size[0] -1

		top		= btm + height
		# if top >= img_size[1] :
		# 	top		= img_size[1] -1

		# width	= right	-left
		# height	= top	-btm

		tmp_annot_list[annot_cnt] = [left, btm, width, height, logit_value]
		# tmp_annot_list[annot_cnt] = tmp_annot

		annot_cnt = annot_cnt + 1
		if annot_cnt == tmp_numface :
			top_file.write("%s\n" %tmp_img_path)
			top_file.write("%d\n" %tmp_numface)
			for k in xrange( len(tmp_annot_list) ) :
				top_file.write("%s %s %s %s %s\n" %( int(tmp_annot_list[k][0]), int(tmp_annot_list[k][1]), int(tmp_annot_list[k][2]), int(tmp_annot_list[k][3]), tmp_annot_list[k][4]) )
