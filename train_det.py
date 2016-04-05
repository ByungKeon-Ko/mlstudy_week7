import numpy as np
import tensorflow as tf
import math

import cvpr_network
import batch_manager
import time
# import Image

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
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25 )

if net_name == '12net' :
	MODE_CONT = 1
	nBatch = 10
	threshold = 0.50
	LearningRate = 0.001
elif net_name == '24net' :
	MODE_CONT = 0
	nBatch = 50
	threshold = 0.50
	LearningRate = 0.01
elif net_name == '48net' :
	MODE_CONT = 1
	# nBatch = 10
	nBatch = 100
	# nBatch = 200
	threshold = 0.50
	# LearningRate = 0.01
	# LearningRate = 0.001
	# LearningRate = 0.0001
	LearningRate = 0.00001

if net_name == '12net' :
	# ckpt_det12 = "tmp/model_12net_2.ckpt"
	ckpt_det12 = "tmp/model_12net_3.ckpt"
	target_ckpt = "tmp/model_12net_3.ckpt"
elif net_name == '24net' :
	ckpt_det12 = "tmp/model_12net_2.ckpt"
	# ckpt_det12 = "tmp/model_12net_3.ckpt"
	# ckpt_det24 = "tmp/model_24net.ckpt"
	# ckpt_det24 = "tmp/model_24net_1.ckpt"		# currently used
	ckpt_det24 = "tmp/model_24net_2.ckpt"
	target_ckpt = ckpt_det24
elif net_name == '48net' :
	ckpt_det12 = "tmp/model_12net_2.ckpt"
	# ckpt_det12 = "tmp/model_12net_3.ckpt"
	# ckpt_det24 = "tmp/model_24net.ckpt"
	ckpt_det24 = "tmp/model_24net_1.ckpt"
	ckpt_det48 = "tmp/model_48net_1.ckpt"
	target_ckpt = ckpt_det48

print "Train DET net start, net : ", net_name
print "MODE_CONT : ", MODE_CONT
print "Learing Rate : ", LearningRate
print "nBatch : ", nBatch
# ############################################################################# #
# ---------------------- NETWORK MODEL ---------------------------------------- #
# ############################################################################# #
BM = batch_manager.BatchManager()
BM.init(net_name)

sess = tf.Session( config=tf.ConfigProto(gpu_options=gpu_options ) )
sess = tf.InteractiveSession()

if net_name == '12net' :
	N12 = cvpr_network.cvpr_12net()
	N12.input_config_noscale()
	N12.infer()
	N12.objective()
	N12.train(LearningRate, threshold)
	network_list = [N12]

elif net_name == '24net' :
	N12 = cvpr_network.cvpr_12net()
	N12.input_config_div2()
	N12.infer()
	N24 = cvpr_network.cvpr_24net()
	N24.input_config_noscale()
	N24.infer()
	N24.objective()
	N24.train(LearningRate, threshold)
	network_list = [N12, N24]

elif net_name == '48net' :
	N12 = cvpr_network.cvpr_12net()
	N12.input_config_div4()
	N12.infer()
	N24 = cvpr_network.cvpr_24net()
	N24.input_config()
	N24.infer()
	N48 = cvpr_network.cvpr_48net()
	N48.infer()
	N48.objective()
	N48.train(LearningRate, threshold)
	network_list = [N12, N24, N48]

init = tf.initialize_all_variables()

saver1 = tf.train.Saver( { 'net12_w_conv1':N12.W_conv1, 'net12_b_conv1':N12.b_conv1, 'net12_w_fc1':N12.W_fc1, 'net12_b_fc1':N12.b_fc1, 'net12_w_fc2':N12.W_fc2, 'net12_b_fc2': N12.b_fc2 } )
if not net_name == '12net' :
	saver2 = tf.train.Saver( { 'net24_w_conv1':N24.W_conv1, 'net24_b_conv1':N24.b_conv1, 'net24_w_fc1':N24.W_fc1, 'net24_b_fc1':N24.b_fc1, 'net24_w_fc2':N24.W_fc2, 'net24_b_fc2': N24.b_fc2 } )
if net_name == '48net' :
	saver3 = tf.train.Saver( { 'net48_w_conv1':N48.W_conv1, 'net48_b_conv1':N48.b_conv1, 'net48_w_conv2':N48.W_conv2, 'net48_b_conv2':N48.b_conv2, 'net48_w_fc1':N48.W_fc1, 'net48_b_fc1':N48.b_fc1, 'net48_w_fc2':N48.W_fc2, 'net48_b_fc2': N48.b_fc2 } )


if net_name == '12net' :
	target_saver = saver1
	if MODE_CONT == 1 :
		saver1.restore(sess, ckpt_det12)
	else :
		sess.run( init )

elif net_name == '24net' :
	target_saver = saver2
	if MODE_CONT == 1 :
		saver1.restore(sess, ckpt_det12)
		saver2.restore(sess, ckpt_det24)
	else :
		sess.run( init )
		saver1.restore(sess, ckpt_det12)

elif net_name == '48net' :
	target_saver = saver3
	if MODE_CONT == 1 :
		saver1.restore(sess, ckpt_det12)
		saver2.restore(sess, ckpt_det24)
		saver3.restore(sess, ckpt_det48)
	else :
		sess.run( init )
		saver1.restore(sess, ckpt_det12)
		saver2.restore(sess, ckpt_det24)

# ############################################################################# #
# ---------------------- FUNCTION LIST ---------------------------------------- #
# ############################################################################# #
def func_loss (net_name, network_list, ccd_12net, ccd_24net, batch ) :
	if net_name == '12net' :
		N12 = network_list[0]
		return N12.cross_entropy.eval(feed_dict={N12.x:batch[0], N12.y_:batch[1] } )
	elif net_name == '24net' :
		N12 = network_list[0]
		N24 = network_list[1]
		return N24.cross_entropy.eval(feed_dict={N24.x:batch[0], N24.y_:batch[1], N24.in_12net: ccd_12net } )
	elif net_name == '48net' :
		N12 = network_list[0]
		N24 = network_list[1]
		N48 = network_list[2]
		return N48.cross_entropy.eval(feed_dict={N48.x:batch[0], N48.y_:batch[1], N48.in_24net: ccd_24net } )

def func_accuracy ( net_name, network_list, ccd_12net, ccd_24net, batch) :
	if net_name == '12net' :
		N12 = network_list[0]
		return N12.accuracy.eval(feed_dict={N12.x:batch[0], N12.y_:batch[1] } )
	elif net_name == '24net' :
		N12 = network_list[0]
		N24 = network_list[1]
		return N24.accuracy.eval(feed_dict={N24.x:batch[0], N24.y_:batch[1], N24.in_12net: ccd_12net } )
	elif net_name == '48net' :
		N12 = network_list[0]
		N24 = network_list[1]
		N48 = network_list[2]
		return N48.accuracy.eval(feed_dict={N48.x:batch[0], N48.y_:batch[1], N48.in_24net: ccd_24net } )

def func_train ( net_name, network_list, ccd_12net, ccd_24net, batch) :
	if net_name == '12net' :
		N12 = network_list[0]
		N12.train_step.run( feed_dict= {N12.x:batch[0], N12.y_: batch[1] } )
		return 1
	elif net_name == '24net' :
		N12 = network_list[0]
		N24 = network_list[1]
		N24.train_step.run( feed_dict= {N24.x:batch[0], N24.y_: batch[1], N24.in_12net: ccd_12net } )
		return 1
	elif net_name == '48net' :
		N12 = network_list[0]
		N24 = network_list[1]
		N48 = network_list[2]
		N48.train_step.run( feed_dict= {N48.x:batch[0], N48.y_: batch[1], N48.in_24net: ccd_24net } )
		return 1


# ############################################################################# #
# ---------------------- TRAIN LOOP    ---------------------------------------- #
# ############################################################################# #
learning_scale = 1
cnt = 0
cnt_epoch = 0
sum_acc = 0
sum_loss = 0
cnt_loss = 0
average_loss = 100
average_acc = 0
ccd_12net = 0
ccd_24net = 0
batch = []
new_epoch_flag = 0
train_accuracy = 0
i = 0
start_time = time.time()

print "loop start!"
while cnt_epoch < 200:
	i = i + 1
	batch = BM.next_batch(nBatch)
	new_epoch_flag = batch[2]

	if not net_name == '12net' :
		ccd_12net = N12.h_fc1.eval( feed_dict={ N12.x:batch[0] } )

	if net_name == '48net' :
		ccd_24net_1 = N24.h_fc1.eval( feed_dict={ N24.x:batch[0], N24.in_12net:ccd_12net } )
		ccd_24net = np.concatenate( [ccd_12net, ccd_24net_1], axis = 3 )

	if (new_epoch_flag == 1) :
		cnt_epoch = cnt_epoch + 1
	
	if i%(1000/nBatch) == 0 :
		loss           = func_loss(net_name, network_list, ccd_12net, ccd_24net, batch )
		train_accuracy = func_accuracy(net_name, network_list, ccd_12net, ccd_24net, batch )
		sum_loss = sum_loss + loss
		sum_acc  = sum_acc  + train_accuracy
		cnt_loss = cnt_loss + 1
		if i%(5000/nBatch) == 0 :
			tmp_loss = sum_loss / float( cnt_loss + 1e-40 )
			tmp_acc  = sum_acc / float( cnt_loss + 1e-40)
			print "step : %d, training accuracy : %g, loss : %g" %(i, tmp_acc, tmp_loss)
			if not math.isnan(tmp_loss) :
				target_saver.save(sess, target_ckpt)

	if (new_epoch_flag == 1) :
		past_loss = average_loss
		average_loss = sum_loss / float( cnt_loss + 1e-40 )
		average_acc  = sum_acc / float( cnt_loss + 1e-40)
		cnt_loss = 0
		sum_loss = 0
		sum_acc  = 0
		print "epoch : %d, step : %d, training accuracy : %g, loss : %g" %(i, cnt_epoch, average_acc, average_loss), (time.time() - start_time)/60.
		start_time = time.time()
		# if average_loss > past_loss :
		# 	LearningRate = LearningRate/10
		# 	print "Reduce Learning Rate : ", LearningRate
		# 	network_list[ len(network_list)-1].train(LearningRate, threshold)
		# 	target_saver.restore(sess, target_ckpt)
		# else :
		# 	target_saver.save(sess, target_ckpt)
		if not math.isnan(average_loss) :
			target_saver.save(sess, target_ckpt)

	func_train (net_name, network_list, ccd_12net, ccd_24net, batch )

if not math.isnan(average_loss) :
	save_path = target_saver.save(sess, target_ckpt)
print "Model saved in file: ", save_path


