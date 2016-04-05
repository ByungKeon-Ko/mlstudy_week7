# 1. Purpose
#	   Non-Maximum Suppression for txt files in output_file folder
#	   for making clear red square
#

import numpy as np
import math
import copy

# ------ Parameters ---------------------- 
base_path	= "/home/bkko/ml_study/week7"
netname = '48net'

if netname == '12net' :
	old_path	= "%s/ofile_12net"		%base_path
	new_path	= "%s/ofile_12net_1"	%base_path
elif netname == '24net' :
	old_path	= "%s/ofile_24net"		%base_path
	new_path	= "%s/ofile_24net_1"	%base_path
elif netname == '48net' :
	old_path	= "%s/ofile_48net"		%base_path
	new_path	= "%s/ofile_48net_1"	%base_path
	
test_list	= "/home/bkko/ml_study/fddb/FDDB-folds/rect-9.txt"

threshold = float(0.80)


# ------ functions ------------------------
def check_neighbor (square1, square2 ) :
	sq1_left = int( square1.split()[0] )
	sq1_top  = int( square1.split()[1] )
	sq1_width  = int( square1.split()[2] )
	sq1_height = int( square1.split()[3] )

	sq2_left = int( square2.split()[0] )
	sq2_top  = int( square2.split()[1] )
	sq2_width  = int( square2.split()[2] )
	sq2_height = int( square2.split()[3] )

	#	cond1 = math.sqrt( abs( sq1_left - sq2_left )**2 + abs( sq1_top - sq2_top )**2 ) <= margin
	#	cond3 = abs( sq1_width  - sq2_width  ) <= margin
	#	cond4 = abs( sq1_height - sq2_height ) <= margin

	# IOU : Intersection Over Union
	sq1_right = sq1_left + sq1_width
	sq1_btm   = sq1_top  - sq1_height
	sq2_right = sq2_left + sq2_width
	sq2_btm   = sq2_top  - sq2_height

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

	cond_iou = IOU >= 0.2

	# Belong & Include
	cond_inc = ( float(A_inter) / float(sq2_size) ) > 0.8
	cond_bel = ( float(A_inter) / float(sq1_size) ) > 0.8

	# cond_inc_1 = (sq1_left <= sq2_left) & (sq1_right >= sq2_right)
	# cond_inc_2 = (sq1_btm  <= sq2_btm ) & (sq1_top   >= sq2_top  )

	# cond_bel_1 = (sq1_left >= sq2_left) & (sq1_right <= sq2_right)
	# cond_bel_2 = (sq1_btm  >= sq2_btm ) & (sq1_top   <= sq2_top  )

	if cond_iou :	  # IOU
		return 1
	# elif cond1 & cond3 & cond4 :  # neighbor for similar size
	#	 return 1
	elif cond_inc :	 # one includes another
		return 1
	elif cond_bel :	 # one belongs to another
		return 1
	else :
		return 0

def check_greater_weight (square1, square2 ) :
	sq1_weight = float(square1.split()[4])
	sq2_weight = float(square2.split()[4])

	if sq1_weight >= sq2_weight :
		return 1
	else :
		return 0


# ----------------------------------------- 
rect_file = open(test_list,  'r')

while 1 :
	rectfile_line = rect_file.readline().rstrip()
	
	cond_eof = rectfile_line == ''
	cond_numface = len(rectfile_line) <= 2
	if cond_numface != 1 :
		cond_newimg = (rectfile_line[0:3] == '200') & (rectfile_line[3]!=' ') & (rectfile_line[3]!='.')
	else :
		cond_newimg = 0

	if cond_eof :	   # end of file
		print "1 set done!!"
		break
	
	elif cond_numface : # the number of face in the image
		pass

	elif cond_newimg :  # new image name
		current_file_name = rectfile_line.split('/')[4]

		org_file = open("%s/out-%s.txt"%(old_path, current_file_name ), 'r')
		# org_file = open("/home/bkko/ml_study/week7/output_file/out-img_375.txt", 'r')
		
		temp_line = org_file.readline().rstrip()   # image name
		temp_line = org_file.readline().rstrip()   # the number of detected faces
		
		squareNum = int(temp_line)
		
		square_list = []
		final_list = []
		
		for i in range(squareNum):
			temp_line = org_file.readline().rstrip()
			temp_weight = temp_line.split()[4]
			if  float(temp_weight) > threshold :
				print "temp weight : ", temp_weight
				square_list.append( temp_line )

		while len(square_list)!=0 :
			temp = square_list.pop()
			flag = 0

			temp_list = copy.deepcopy( square_list )

			for i in range( len(temp_list) ) :
				temp2 = temp_list.pop()
				cond_near = check_neighbor(temp, temp2)

				if cond_near :
					cond_w = check_greater_weight(temp, temp2)
					if cond_w :
						square_list.remove(temp2)
						pass
					else :
						flag = 1
						break

			if flag==0 :
				final_list.insert(0, temp)
		
		print "done!!"
		
		new_file = open("%s/new-%s.txt"%(new_path, rectfile_line.split('/')[4]), 'w')
		
		org_file = open("%s/out-%s.txt"%(old_path, current_file_name ), 'r')
		temp_line = org_file.readline().rstrip()   # image name
		new_file.write("%s\n"%temp_line)
		temp_line = org_file.readline().rstrip()   # the number of detected faces
		finalNum = len(final_list)
		new_file.write("%d\n"%finalNum)
		
		for i in range(finalNum):
			new_file.write("%s\n"%final_list.pop())


