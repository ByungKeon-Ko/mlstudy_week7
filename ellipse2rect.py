# 1. Purpose
#       read FDDB-fold-##_ellipse.txt file and
#       and generate ##_rectangle.txt file
#   
# 2. Target Data sets
#       01~08 : training set, 09~10 : test set
#       
# 3. Ellipse annotations
#       < major_axis_radius,  minor_axis_radius, angle, center_x, center_y 1>
#       
# 4. Rectangular annotations
#       < left_x top_y width height detection_score >
#         ** detection_score is T.B.D

import numpy as np
import math

# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/fddb/FDDB-folds"
# fold_num = 9
fold_num = 10

# ------ functions ------------------------
def convert_shape( major_axis, minor_axis, angle, center_x, center_y ):
    th = math.radians(angle)
    height_1 = 2 * major_axis * math.cos(th)
    width_1  = 2 * major_axis * math.sin(th)
    height_2 = 2 * minor_axis * math.sin(th)
    width_2  = 2 * minor_axis * math.cos(th)

    height = int( max( height_1, height_2 ) )
    width  = int( max( width_1,  width_2  ) )

    left_x = int( center_x - width / 2 )
    # top_y  = int( center_y + height/ 2 )
    btm_y  = int( center_y - height/ 2 )

    # return (left_x, top_y, width, height, 0)
    return (left_x, btm_y, width, height, 0)

# ----------------------------------------- 

print "ellipse2rect.py start !!"

for i in xrange(fold_num, fold_num+1) :
# for i in xrange(1, fold_num+1) :
    # ellipse_file = open("%s/FDDB-fold-0%s-ellipseList.txt"%(base_path, i), 'r')
    ellipse_file = open("%s/FDDB-fold-%s-ellipseList.txt"%(base_path, i), 'r')
    rectangle_file = open("%s/rectbtm-%s.txt" %(base_path, i), 'w')

    while 1 :
        temp_line = ellipse_file.readline().rstrip()

        cond_eof = temp_line == ''
        cond_numface = len(temp_line) <= 2
        if cond_numface != 1 :
            cond_newimg = (temp_line[0:3] == '200') & (temp_line[3]!=' ') & (temp_line[3]!='.')
        else :
            cond_newimg = 0

        if cond_eof :
            break
        elif cond_newimg :   # new image
            rectangle_file.write("%s\n"%temp_line)
        elif cond_numface :     # the number of face in the image
            rectangle_file.write("%s\n"%temp_line)
        else :                        # ellipse information
            temp_line = temp_line.split()
            major_axis = float( temp_line[0] )
            minor_axis = float( temp_line[1] )
            angle      = float( temp_line[2] )
            center_x   = float( temp_line[3] )
            center_y   = float( temp_line[4] )
            rectangle_file.write("%d %d %d %d %d\n" %convert_shape( major_axis, minor_axis, angle, center_x, center_y ) )
    rectangle_file.close()
    ellipse_file.close()

