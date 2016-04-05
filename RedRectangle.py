# 1. Purpose
#	read rect-#.txt file and generate new images with red rectangles

import numpy as np
import Image
import copy

# ------ Parameters ---------------------- 
# dataset_name = 'aflw'
dataset_name = 'fddb'
btm_annot_mode = 1

# test_list	= "/home/bkko/ml_study/fddb/FDDB-folds/rect-9.txt"
test_list	= "/home/bkko/ml_study/week7/ofile_12net/o_annot.txt"

if dataset_name == 'aflw' :
	img_path =  "/home/bkko/ml_study/aflw/aflw/data/flickr"
elif dataset_name == 'fddb' :
	img_path =  "/home/bkko/ml_study/fddb"

text_path	= "/home/bkko/ml_study/week7/ofile_12net"
new_path = "./output_image"

# ------ functions ------------------------
def draw_rectangle (im_matrix, left_x, top_y, width, height, btm_annot_mode) :
	row, col, colorNum = im_matrix.shape

	if btm_annot_mode == 1 :
		btm_y = top_y
		top_y = btm_y + height
	else :
		btm_y = top_y - height

	right_x = left_x + width

	if(right_x >= col) :
		right_x = col -1
	if left_x < 0 :
		left_x = 0
	if top_y >= row :
		top_y = row-1
	if btm_y < 0 :
		btm_y = 0

	for i in xrange(left_x, right_x+1) :
		im_matrix[top_y, i] = [255,0,0]
		im_matrix[btm_y, i] = [255,0,0]
	for j in xrange(btm_y, top_y+1) :
		im_matrix[j, left_x  ] = [255,0,0]
		im_matrix[j, right_x ] = [255,0,0]

# ----------------------------------------- 
fold_file = open(test_list,  'r')

new_image_name = -1

while 1 :
	text_line = fold_file.readline().rstrip()
	
	cond_eof = text_line == ''
	cond_numface = ( len( text_line.split('/') ) == 1 ) & ( len( text_line.split(' ') ) == 1 )
	if cond_numface != 1 :
		cond_newimg = len( text_line.split('/') ) > 1
	else :
		cond_newimg = 0
	
	if cond_eof :	# end of file
		print "eof!"
		new_image = Image.fromarray(im_matrix)
		new_image.save(new_image_name)
		# new_image.show()
		break
	
	elif cond_numface : # the number of face in the image
		pass
	
	elif cond_newimg :  # new image name
		print "new image!: ", text_line
		if new_image_name != -1 :
			new_image = Image.fromarray(im_matrix)
			new_image.save(new_image_name)
			# new_image.show()
		
		if dataset_name == 'aflw' :
			old_image = Image.open("%s/%s" %(img_path, text_line), 'r')
		elif dataset_name == 'fddb' :
			old_image = Image.open("%s/%s.jpg" %(img_path, text_line), 'r')
		col, row = old_image.size
		pixels = old_image.load()
		
		tmp_matrix = np.asarray(old_image)
		im_matrix = copy.deepcopy(tmp_matrix)
		
		if dataset_name == 'aflw' :
			img_name = text_line.split('/')[1]
		elif dataset_name == 'fddb' :
			img_name = text_line.split('/')[4]
		new_image_name = "%s/table_det48_%s.jpg" %(new_path, img_name)
	else :
		ell_line = text_line.split()
		left_x =  int( ell_line[0] )
		width  =  int( ell_line[2] )
		height =  int( ell_line[3] )
		if btm_annot_mode :
			btm_y  =  int( ell_line[1] )
			draw_rectangle(im_matrix, left_x, btm_y, width, height, btm_annot_mode)
		else :
			top_y  =  int( ell_line[1] )
			draw_rectangle(im_matrix, left_x, top_y, width, height, btm_annot_mode)





