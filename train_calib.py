
import numpy as np
import tensorflow as tf

import cvpr_network
import batch_manager
import math
import time

# ------ Parameters ---------------------- 
base_path = "/home/bkko/ml_study/week7"

# net_name = '12net'
# net_name = '24net'
net_name = '48net'

if net_name == '12net' :
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.02 )
elif net_name == '24net' :
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05 )
elif net_name == '48net' :
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15 )

if net_name == '12net' :
	MODE_CONT = 1
	# nBatch = 10
	# LearningRate = 2.0
	LearningRate = 0.001
	nBatch = 100
elif net_name == '24net' :
	MODE_CONT = 1
	# nBatch = 10
	LearningRate = 0.1
	nBatch = 50
	# LearningRate = 2.0
elif net_name == '48net' :
	MODE_CONT = 1
	nBatch = 10
	# LearningRate = 0.001
	# LearningRate = 1e-4
	# LearningRate = 1e-5
	LearningRate = 1e-1
	# nBatch = 50
	# LearningRate = 2.0

# start 	 --> nBatch = 50, LearningRate : 30
# loss = 0.5 --> LearningRate : 3
# loss = 0.4 --> nBatch = 200, LearningRate : 1

if net_name == '12net' :
	# ckpt_file = "tmp/model_calib12_3.ckpt"
	ckpt_file = "tmp/model_calib12_4.ckpt"
elif net_name == '24net' :
	ckpt_file = "tmp/model_calib24.ckpt"
elif net_name == '48net' :
	ckpt_file = "tmp/model_calib48.ckpt"

print "Calib net training : ", net_name, LearningRate
print "continous mode? ", MODE_CONT

# ----------------------------------------- 
BM = batch_manager.BatchManager_calib()
BM.init(net_name)

sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

if net_name == '12net' :
	CALNET = cvpr_network.cvpr_calib_12net()
elif net_name == '24net' :
	CALNET = cvpr_network.cvpr_calib_24net()
elif net_name == '48net' :
	CALNET = cvpr_network.cvpr_calib_48net()

CALNET.input_config_noscale()
CALNET.infer()
CALNET.objective()
CALNET.train(LearningRate)

# ---- Network modeling ------------------------------------------------------------------- #

if net_name == '12net' :
	saver = tf.train.Saver( { 
		'cal12_w_conv1':CALNET.W_conv1, 'cal12_b_conv1':CALNET.b_conv1,
		'cal12_w_fc1':CALNET.W_fc1, 'cal12_b_fc1':CALNET.b_fc1,
		'cal12_w_fc2':CALNET.W_fc2, 'cal12_b_fc2': CALNET.b_fc2 } )
elif net_name == '24net' :
	saver = tf.train.Saver( { 
		'cal24_w_conv1':CALNET.W_conv1, 'cal24_b_conv1':CALNET.b_conv1,
		'cal24_w_fc1':CALNET.W_fc1, 'cal24_b_fc1':CALNET.b_fc1,
		'cal24_w_fc2':CALNET.W_fc2, 'cal24_b_fc2': CALNET.b_fc2 } )
elif net_name == '48net' :
	saver = tf.train.Saver( { 
		'cal48_w_conv1':CALNET.W_conv1, 'cal48_b_conv1':CALNET.b_conv1,
		'cal48_w_conv2':CALNET.W_conv2, 'cal48_b_conv2':CALNET.b_conv2,
		'cal48_w_fc1':CALNET.W_fc1, 'cal48_b_fc1':CALNET.b_fc1,
		'cal48_w_fc2':CALNET.W_fc2, 'cal48_b_fc2': CALNET.b_fc2 } )

if MODE_CONT==1 :
	saver.restore(sess, ckpt_file)
else :
	init = tf.initialize_all_variables()
	sess.run( init )

cnt = 0
cnt_epoch = 0
sum_acc = 0
sum_loss = 0
cnt_loss = 0
average_loss = 100
average_acc = 0
y_conv = 0
batch = []
new_epoch_flag = 0
train_accuracy = 0
cnt_loss = 0

# for i in xrange(50000):
i = 0
confusion_matrix = np.zeros([45,45])
start_time = time.time()

while cnt_epoch < 1000:
	i = i + 1
	batch = BM.next_batch(nBatch)
	new_epoch_flag = batch[2]

	if (new_epoch_flag == 1) :
		cnt_epoch = cnt_epoch + 1

	# y_conv = CALNET.y_conv_reshape.eval( feed_dict={CALNET.x:batch[0]} )
	# if np.max(y_conv) == 1 :
	# 	print "overflow protect!! skip 1 loop"
	# 	continue

	if (i%100)==0 :
		loss = CALNET.cross_entropy.eval(feed_dict={CALNET.x:batch[0], CALNET.y_:batch[1] } )
		train_accuracy = CALNET.accuracy.eval(feed_dict={CALNET.x:batch[0], CALNET.y_:batch[1] } )
		sum_loss = sum_loss + loss
		sum_acc = sum_acc + train_accuracy
		cnt_loss = cnt_loss + 1

		if i%500 == 0 :
			tmp_loss = sum_loss / float( cnt_loss + 1e-40 )
			tmp_acc  = sum_acc / float( cnt_loss + 1e-40)
			print "step : %d, epoch : %d, training accuracy : %g, loss : %g" %(i, cnt_epoch, tmp_acc, tmp_loss)
			if not math.isnan(tmp_loss) :
				saver.save(sess, ckpt_file)
		# y_conv = CALNET.y_conv_reshape.eval( feed_dict={CALNET.x:batch[0]} )
		# for k in xrange(nBatch) :
		# 	tmp_class = np.argmax(y_conv[k])
		# 	org_class = np.argmax(batch[1][k])
		# 	confusion_matrix[org_class, tmp_class] = confusion_matrix[org_class, tmp_class] + 1

	if (new_epoch_flag == 1):
		past_loss = average_loss
		average_loss = sum_loss / float( cnt_loss +1e-40) * 10
		average_acc  = sum_acc / float( cnt_loss + 1e-40)
		sum_loss = 0
		sum_acc = 0
		cnt_loss = 0

		# test_yselect = CALNET.y_select.eval( feed_dict={CALNET.x:batch[0]} )
		# print test_yselect[0:10]

		print "epoch : %d, training accuracy : %g, loss : %g" %(cnt_epoch, average_acc, average_loss), (time.time() - start_time)/60.
		start_time = time.time()
		# if average_loss > past_loss :
		# 	LearningRate = LearningRate/10
		# 	print "Reduce Learning Rate : ", LearningRate
		# 	CALNET.train(LearningRate)
		# 	saver.restore(sess, ckpt_file)
		# else :
		if not math.isnan(average_loss) :
			save_path = saver.save(sess, ckpt_file)

		# if cnt_epoch %20 == 0 :
		# 	for k in xrange(45) :
		# 		print "MAT1-%s : "%k, confusion_matrix[k][0:10]
		# 	for k in xrange(45) :
		# 		print "MAT2-%s : "%k, confusion_matrix[k][11:20]
		# 	for k in xrange(45) :
		# 		print "MAT3-%s : "%k, confusion_matrix[k][21:30]
		# 	for k in xrange(45) :
		# 		print "MAT4-%s : "%k, confusion_matrix[k][31:40]
		# 	for k in xrange(45) :
		# 		print "MAT5-%s : "%k, confusion_matrix[k][41:45]
		# 	confusion_matrix = np.zeros([45,45])

	CALNET.train_step.run( feed_dict= {CALNET.x:batch[0], CALNET.y_: batch[1] } )

	# print "part4 : ", time.time() - start_time
	# start_time = time.time()

if not math.isnan(average_loss) :
	save_path = saver.save(sess, ckpt_file)
print "Model saved in file: ", save_path

