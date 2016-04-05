
# index : 0~44
def mapping_calib_para ( index ) :
	# list_s = [ 0.83, 0.91, 1.0, 1.10, 1.21 ]
	# list_x = [ -0.17, 0.0, 0.17 ]
	# list_y = [ -0.17, 0.0, 0.17 ]
	index_s = ( int(index)/9 ) %5
	index_x = ( int(index)/3 ) %3
	index_y = ( int(index)   ) %3
	return index_s, index_x, index_y

def decoding_calib ( s, x, y ) :
	list_s = [ 0.83, 0.91, 1.0, 1.10, 1.21 ]
	list_x = [ -0.17, 0.0, 0.17 ]
	list_y = [ -0.17, 0.0, 0.17 ]
	return float(list_s[s]), float(list_x[x]), float(list_y[y])

def convert_invcalib ( sq_old, s, x, y ) :
	left_x	= float( sq_old[0] )
	btm_y	= float( sq_old[1] )
	width	= float( sq_old[2] )
	height	= float( sq_old[3] )

	new_left_x	= left_x + x * width
	new_btm_y	= btm_y + y * height
	new_width	= width * s
	new_height	= height * s

	new_square = [ new_left_x, new_btm_y, new_width, new_height]
	return new_square

def transform_calib ( square, s, x, y ) :
	left_x	= float( square[0] )
	btm_y	= float( square[1] )
	width	= float( square[2] )
	height	= float( square[3] )

	new_left_x	= left_x - x * width/s
	new_btm_y	= btm_y - y * height/s
	new_width	= width / s
	new_height	= height / s

	new_square = [ new_left_x, new_btm_y, new_width, new_height]
	return new_square

