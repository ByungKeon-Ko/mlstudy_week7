import tensorflow as tf
import Image
import numpy as np
import random
import PIL
from PIL import ImageOps
import sys
import time

# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/week7"

MinFace = 48

# ----------------------------------------- 

rand_index = 0
img_index = 0

class data_augment () :
	def init (self,net_width) :
		self.x_image	= tf.placeholder (tf.float32, [net_width, net_width, 3], name = 'x' )
		x_1		= tf.image.random_flip_left_right ( self.x_image )
		# x_2		= tf.image.random_hue ( x_1, max_delta=0.1 )
		x_2		= tf.image.random_hue ( x_1, max_delta=0.05 )
		x_3		= tf.image.random_contrast ( x_2, lower=0.8, upper = 1.0 )
		self.y_image	= tf.image.random_saturation ( x_3, lower = 0.8, upper = 1.0 )
		# self.y_image	= tf.image.random_flip_left_right ( self.x_image )

class BatchManager ( ) :
	def init (self, netname):
		if netname == '12net' :
			self.net_width = 12
		elif netname == '24net' :
			self.net_width = 24
		elif netname == '48net' :
			self.net_width = 48

		# self.psNum = 24384
		self.psNum = 24383
		if netname == '48net' :
			self.nsNum = 140000
		else :
			self.nsNum = 200000
			# self.nsNum = 100000
		if netname == '48net' :
			# self.calNum = 200000
			self.calNum = 10
			# self.calNum = 1090000	# cannot be loaded at once
			self.aflwNum = 214000
		elif netname == '24net' :
			self.calNum = 200000
			self.aflwNum = 214000
		else :
			self.calNum = 50000
			self.aflwNum = 100000

		self.ps_max_index = self.psNum
		self.ns_max_index = self.nsNum
		self.cal_max_index = self.calNum
		self.aflw_max_index = self.aflwNum
		self.cnt_in_epoch = 0
		self.dataset_index = 0

		self.DATA_AUG = data_augment()
		self.DATA_AUG.init(self.net_width)

		# prepare negative sample
		self.ns_matrix = np.zeros( [self.nsNum, self.net_width, self.net_width, 3] ).astype(np.uint8)
		for i in xrange(self.ns_max_index) :
			if i%50000 ==0 :
				print "NS :", i
			if netname == '12net' :
				tmp_matrix = np.asarray( Image.open("%s/NS_det12/ns-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			elif netname == '24net' :
				tmp_matrix = np.asarray( Image.open("%s/NS_det24_1/ns-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			elif netname == '48net' :
				tmp_matrix = np.asarray( Image.open("%s/NS_det48_2/ns-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			self.ns_matrix[i] = tmp_matrix

		self.ns_index_list = range(self.ns_max_index)

		# prepare positive sample
		self.ps_matrix = np.zeros( [self.psNum, self.net_width, self.net_width, 3] ).astype(np.uint8)
		for i in xrange(self.ps_max_index) :
			if i %50000 == 0 :
				print "PS :", i
			# tmp_matrix = np.asarray( Image.open("%s/PS_aflw/ps-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8).reshape(MinFace, MinFace, 3 )
			if netname == '12net' :
				tmp_matrix = np.asarray( Image.open("%s/PS_det12/ps-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			elif netname == '24net' :
				tmp_matrix = np.asarray( Image.open("%s/PS_det24/ps-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			elif netname == '48net' :
				tmp_matrix = np.asarray( Image.open("%s/PS_det48/ps-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			self.ps_matrix[i] = tmp_matrix

		self.ps_index_list = range(self.ps_max_index)

		# prepare calibration sample ( kind of data augmentation : shift, resize )
		self.cal_matrix = np.zeros( [self.calNum, self.net_width, self.net_width, 3] ).astype(np.uint8)
		# self.load_ps_cal()

		# prepare aflwibration sample ( kind of data augmentation : shift, resize )
		self.aflw_matrix = np.zeros( [self.aflwNum, self.net_width, self.net_width, 3] ).astype(np.uint8)
		for i in xrange(self.aflw_max_index) :
			if i %50000 == 0 :
				print "NS_AFLW :", i
			if netname == '24net' :
				tmp_matrix = np.asarray( Image.open("%s/NS_aflw24/ns-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			elif netname == '48net' :
				tmp_matrix = np.asarray( Image.open("%s/NS_aflw48_1/ns-%d.jpg" %(base_path, i), 'r' ) ).astype(np.uint8)
			self.aflw_matrix[i] = tmp_matrix

		self.aflw_index_list = range(self.aflw_max_index)

	def load_ps_cal (self):
		k = 0
		for i in xrange(self.cal_max_index) :
			k = k + random.randint(1,9)
			if i %50000 == 0 :
				print "CAL :", i
			if self.net_width == 12 :
				tmp_matrix = np.asarray( Image.open("%s/PS_cal12/ps-%d.jpg" %(base_path, k), 'r' ) ).astype(np.uint8)
			elif self.net_width == 24 :
				tmp_matrix = np.asarray( Image.open("%s/PS_cal24/ps-%d.jpg" %(base_path, k), 'r' ) ).astype(np.uint8)
			elif self.net_width == 48 :
				tmp_matrix = np.asarray( Image.open("%s/PS_cal48/ps-%d.jpg" %(base_path, k), 'r' ) ).astype(np.uint8)
			self.cal_matrix[i] = tmp_matrix

		self.cal_index_list = range(self.cal_max_index)

	def next_batch (self, nBatch):
		x_batch = np.zeros([nBatch, self.net_width, self.net_width, 3]).astype('float32')
		y_batch = np.zeros([nBatch, 1]).astype('uint8')

		self.cnt_in_epoch = self.cnt_in_epoch + nBatch
		new_epoch_flag = 0
		# if ( self.ps_max_index <= nBatch ) | ( self.ns_max_index <= nBatch ) | ( self.cal_max_index <= nBatch ) | ( self.aflw_max_index <= nBatch ):
		if ( self.ps_max_index <= nBatch ) | ( self.ns_max_index <= nBatch ) | ( self.aflw_max_index <= nBatch ):
			# print "Reset Batch Manager "
			self.cnt_in_epoch = 0
			new_epoch_flag = 1
			self.ps_max_index = self.psNum
			self.ns_max_index = self.nsNum
			self.cal_max_index = self.calNum
			self.aflw_max_index = self.aflwNum
			self.ps_index_list = range(self.ps_max_index)
			self.ns_index_list = range(self.ns_max_index)
			self.cal_index_list = range(self.cal_max_index)
			self.aflw_index_list = range(self.aflw_max_index)
			# self.load_ps_cal()

		for i in xrange(nBatch) :
			if self.net_width == 12 : 
				if random.randint(0,5) >= 1 :
					x_batch[i], y_batch[i] = self.ns_batch()
				else :
					if random.randint(0,2) >= 2 :
						x_batch[i], y_batch[i] = self.ps_batch()
					else :
						x_batch[i], y_batch[i] = self.cal_batch()

			elif self.net_width == 24 : 
				if random.randint(0,1) >= 1 :
					if random.randint(0,1) == 1 :
						x_batch[i], y_batch[i] = self.ns_batch()
					else :
						x_batch[i], y_batch[i] = self.aflw_batch()
				else :
					x_batch[i], y_batch[i] = self.cal_batch()

			elif self.net_width == 48 : 
				# if random.randint(0,4) >= 4 :
				# if random.randint(0,200) >= 200 :
				# if random.randint(0,5) >= 1 :
				if random.randint(0,20) >= 1 :
					if random.randint(0,30) >= 20 :
						x_batch[i], y_batch[i] = self.ns_batch()
					else :
						x_batch[i], y_batch[i] = self.aflw_batch()
				else :
					x_batch[i], y_batch[i] = self.ps_batch()
					# x_batch[i], y_batch[i] = self.cal_batch()
					# if random.randint(0,2) >= 1 :
					# 	x_batch[i], y_batch[i] = self.ps_batch()
					# else :
					# 	x_batch[i], y_batch[i] = self.cal_batch()




		# x_batch = np.reshape(x_batch, [nBatch, MinFace*MinFace*3] ) + np.random.randn(nBatch, MinFace*MinFace*3) * 4.0/255.0
		x_batch = np.reshape(x_batch, [nBatch, self.net_width*self.net_width*3] )

		return [x_batch, y_batch, new_epoch_flag]

	def ps_batch (self):
		x_batch = np.zeros([self.net_width, self.net_width, 3]).astype('float32')
		y_batch = np.zeros([1]).astype('uint8')

		rand_index = self.ps_index_list.pop( random.randint(0, self.ps_max_index-1)     )

		# org_file = org_file.rotate(random.randint(-20, 20) )
		org_matrix = self.DATA_AUG.y_image.eval( feed_dict={self.DATA_AUG.x_image:self.ps_matrix[rand_index]} )
		# org_matrix = self.ps_matrix[rand_index]

		x_batch = np.divide( org_matrix, 255.0 )

		y_batch = np.ones([1])
		self.ps_max_index = self.ps_max_index -1
		return [x_batch, y_batch]

	def cal_batch (self):
		x_batch = np.zeros([self.net_width, self.net_width, 3]).astype('float32')
		y_batch = np.zeros([1]).astype('uint8')

		rand_index = self.cal_index_list.pop( random.randint(0, self.cal_max_index-1)     )

		# org_file = org_file.rotate(random.randint(-20, 20) )
		org_matrix = self.DATA_AUG.y_image.eval( feed_dict={self.DATA_AUG.x_image:self.cal_matrix[rand_index]} )
		# org_matrix = self.cal_matrix[rand_index]

		x_batch = np.divide( org_matrix, 255.0 )

		y_batch = np.ones([1])
		self.cal_max_index = self.cal_max_index -1
		return [x_batch, y_batch]

	def ns_batch (self):
		x_batch = np.zeros([self.net_width, self.net_width, 3]).astype('float32')
		y_batch = np.zeros([1]).astype('uint8')

		rand_index = self.ns_index_list.pop( random.randint(0, self.ns_max_index-1) )

		# org_file = org_file.rotate(random.randint(-20, 20) )
		org_matrix = self.DATA_AUG.y_image.eval( feed_dict={self.DATA_AUG.x_image:self.ns_matrix[rand_index]} )
		# org_matrix = self.ns_matrix[rand_index]

		x_batch = np.divide( org_matrix, 255.0 )

		y_batch = np.zeros([1])
		self.ns_max_index = self.ns_max_index -1
		return [x_batch, y_batch]

	def aflw_batch (self):
		x_batch = np.zeros([self.net_width, self.net_width, 3]).astype('float32')
		y_batch = np.zeros([1]).astype('uint8')

		rand_index = self.aflw_index_list.pop( random.randint(0, self.aflw_max_index-1)     )

		# org_file = org_file.rotate(random.randint(-20, 20) )
		org_matrix = self.DATA_AUG.y_image.eval( feed_dict={self.DATA_AUG.x_image:self.aflw_matrix[rand_index]} )
		# org_matrix = self.aflw_matrix[rand_index]

		x_batch = np.divide( org_matrix, 255.0 )

		y_batch = np.zeros([1])
		self.aflw_max_index = self.aflw_max_index -1
		return [x_batch, y_batch]


class BatchManager_calib ( ) :
	def init (self, netname):
		if netname == '12net' :
			self.net_width = 12
		elif netname == '24net' :
			self.net_width = 24
		elif netname == '48net' :
			self.net_width = 48

		if netname == '12net' :
			self.psNum = 24300 * 45
		if netname == '24net' :
			self.psNum = 24300 * 45
		else :
			# self.psNum = 12300 * 45
			self.psNum = 24300 * 45

		self.ps_max_index = self.psNum
		self.cnt_in_epoch = 0

		# prepare data
		self.ps_matrix = []
		for i in xrange(self.ps_max_index) :
			if i %100000 == 0 :
				print "Data :", i
			tmp_matrix = np.asarray( Image.open("%s/PS_cal%s/ps-%d.jpg" %(base_path, self.net_width, i), 'r' ) ).astype(np.uint8)

			self.ps_matrix.append( tmp_matrix )

		# prepare label
		self.label_matrix = []
		annot_file = open("%s/PS_cal%s/calib_annot.txt" %(base_path, self.net_width), 'r')
		line_matrix = annot_file.readlines()
		for i in xrange(self.ps_max_index) :
			if i %100000 == 0 :
				print "Label :", i
			tmp = line_matrix[i].rstrip().split(',')[1:4]
			one_hot_encoding = int(tmp[0])*9 + int(tmp[1])*3 + int(tmp[2])
			label = np.zeros([45] ).astype(np.float16)
			label[one_hot_encoding] = 1
			self.label_matrix.append ( label )

		self.ps_index_list = range(self.ps_max_index)

	def next_batch (self, nBatch):
		x_batch = np.zeros([nBatch, self.net_width, self.net_width, 3]).astype('float32')
		y_batch = np.zeros([nBatch, 45]).astype('uint8')

		self.cnt_in_epoch = self.cnt_in_epoch + nBatch
		new_epoch_flag = 0
		if ( self.ps_max_index <= nBatch ) :
			# print "Reset Batch Manager "
			self.cnt_in_epoch = 0
			new_epoch_flag = 1
			self.ps_max_index = self.psNum
			self.ps_index_list = range(self.ps_max_index)

		for i in xrange(nBatch) :
			x_batch[i], y_batch[i] = self.ps_batch()

		# x_batch = np.reshape(x_batch, [nBatch, MinFace*MinFace*3] ) + np.random.randn(nBatch, MinFace*MinFace*3) * 4.0/255.0
		x_batch = np.reshape(x_batch, [nBatch, self.net_width*self.net_width*3] )

		return [x_batch, y_batch, new_epoch_flag]

	def ps_batch (self):
		x_batch = np.zeros([self.net_width, self.net_width, 3]).astype('float32')
		y_batch = np.zeros([1]).astype('uint8')

		rand_index = self.ps_index_list.pop( random.randint(0, self.ps_max_index-1)     )
		# org_file = org_file.rotate(random.randint(-20, 20) )
		# org_matrix = self.DATA_AUG.y_image.eval( feed_dict={self.DATA_AUG.x_image:self.ps_matrix[rand_index]} )
		org_matrix = self.ps_matrix[rand_index]
		x_batch = np.divide( org_matrix, 255.0 )

		y_batch = self.label_matrix[rand_index]
		self.ps_max_index = self.ps_max_index -1
		return [x_batch, y_batch]

