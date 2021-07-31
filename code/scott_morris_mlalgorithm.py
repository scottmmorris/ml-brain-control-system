import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import time
import argparse
import pyedflib

##########################################################################################
# Model Information
##########################################################################################

conv_1_shape = '3*3*1*32'
pool_1_shape = '2*2'

conv_2_shape = 'None'
pool_2_shape = 'None'

conv_3_shape = 'None'
pool_3_shape = 'None'

conv_4_shape = 'None'
pool_4_shape = 'None'

n_person = 2
window_size = 10

calibration = 'N'
norm_type='2D'
regularization_method = 'dropout'

##########################################################################################
# Pre-process Dataset
##########################################################################################

# id of begin subject
begin_subject = 1

# id of end subject
end_subject = 108

# set the dataset directory
dataset_dir = "d:/ToRNeuralNetork/Cascade-Parallel-master/1D_CNN/raw_data/"

# define an output file
output_file = "conv_"+str(begin_subject)+"_"+str(end_subject)+"_eeg_summary_075_train_posibility_roc"

# function to receive arguments for key parameters
def get_args():
	parser = argparse.ArgumentParser()
	
	hpstr = "set dataset directory"
	parser.add_argument('-d', '--directory', default="d:/ToRNeuralNetork/Cascade-Parallel-master/raw_data/", nargs='*', type=str, help=hpstr)

	hpstr = "set window size"
	parser.add_argument('-w', '--window', default=10, nargs='*', type=int, help=hpstr)

	hpstr = "set begin person"
	parser.add_argument('-b', '--begin', default=1, nargs='?', type=int, help=hpstr)

	hpstr = "set end person"
	parser.add_argument('-e', '--end', default=2, nargs='?', type=int, help=hpstr)

	hpstr = "set output directory"
	parser.add_argument('-o', '--output_dir', default="d:/ToRNeuralNetork/Cascade-Parallel-master/", nargs='*', help=hpstr)

	hpstr = "set whether store data"
	parser.add_argument('--set_store', action='store_true', help=hpstr)

	args = parser.parse_args()
	return(args)

# print values of the key parameters
def print_top(dataset_dir, window_size, begin_subject, end_subject, output_dir, set_store):
	print("######################## PhysioBank EEG data preprocess ########################	\
		   \n#### Author: Dalin Zhang	UNSW, Sydney	email: zhangdalin90@gmail.com #####	\
		   \n# input directory:	%s \
		   \n# window size:		%d 	\
		   \n# begin subject:	%d 	\
		   \n# end subject:		%d 	\
		   \n# output directory:	%s	\
		   \n# set store:		%s 	\
		   \n##############################################################################"% \
			(dataset_dir,	\
			window_size,	\
			begin_subject,	\
			end_subject,	\
			output_dir,		\
			set_store))
	return None

def read_data(file_name):
	f = pyedflib.EdfReader(file_name)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
	    sigbufs[i, :] = f.readSignal(i)
	sigbuf_transpose = np.transpose(sigbufs)
	signal = np.asarray(sigbuf_transpose)
	signal_labels = np.asarray(signal_labels)
	f._close()
	del f
	return signal, signal_labels

def data_1Dto2D(data, Y=10, X=11):
	data_2D = np.zeros([Y, X])
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0, data[21], data[22], data[23], 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0, data[24], data[25], data[26], data[27], data[28], 	   	 0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0, data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37], 	 	 0) 
	data_2D[3] = (	  	 0, data[38],  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6], data[39], 		 0) 
	data_2D[4] = (data[42], data[40],  data[7],  data[8],  data[9], data[10], data[11], data[12], data[13], data[41], data[43]) 
	data_2D[5] = (	  	 0, data[44], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[45], 		 0) 
	data_2D[6] = (	  	 0, data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54], 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0, data[55], data[56], data[57], data[58], data[59], 	   	 0, 	   0, 		 0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0, data[60], data[61], data[62], 	   0, 	   	 0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0, data[63], 		 0, 	   0, 	   	 0, 	   0, 		 0) 
	return data_2D

def norm_dataset(dataset_1D):
	norm_dataset_1D = np.zeros([dataset_1D.shape[0], 64])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
	return norm_dataset_1D

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data.nonzero()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def dataset_1Dto2D(dataset_1D):
	dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		dataset_2D[i] = data_1Dto2D(dataset_1D[i])
	return dataset_2D

def norm_dataset_1Dto2D(dataset_1D):
	norm_dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
	return norm_dataset_2D

def windows(data, size):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += (size/2)

def segment_signal_without_transition(data, label, window_size):
	for (start, end) in windows(data, window_size):
		if((len(data[start:end]) == window_size) and (len(set(label[start:end]))==1)):
			if(start == 0):
				segments = data[start:end]
				# labels = stats.mode(label[start:end])[0][0]
				labels = np.array(list(set(label[start:end])))
			else:
				segments = np.vstack([segments, data[start:end]])
				labels = np.append(labels, np.array(list(set(label[start:end]))))
				# labels = np.append(labels, stats.mode(label[start:end])[0][0])
	return segments, labels

# function that mixes all the data and labels
def apply_mixup(dataset_dir, window_size, start=1, end=110):
	# initial empty label arrays
	label_inter	= np.empty([0])
	# initial empty data array
	data_inter	= np.empty([0, window_size, 10, 11])
	for j in range(start, end):
		if (j == 89):
			j = 109
		# get directory name for one subject
		data_dir = dataset_dir+"S"+format(j, '03d')
		# get task list for one subject
		task_list = [task for task in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, task))]
		for task in task_list:
			if(("R02" in task) or ("R04" in task) or ("R06" in task)): # R02: eye closed; R04, R06: motor imagery tasks
				print(task+" begin:")
				# get data file name and label file name
				data_file 	= data_dir+"/"+task+"/"+task+".csv"
				label_file 	= data_dir+"/"+task+"/"+task+".label.csv"
				# read data and label
				data		= pd.read_csv(data_file)
				label		= pd.read_csv(label_file)
				# remove rest label and data during motor imagery tasks
				data_label	= pd.concat([data, label], axis=1)
				data_label	= data_label.loc[data_label['labels']!= 'rest']
				# get new label
				label		= data_label['labels']
				# get new data and normalize
				data_label.drop('labels', axis=1, inplace=True)
				data		= data_label.to_numpy()
				data		= norm_dataset(data)
				# convert 1D data to 2D
				data		= dataset_1Dto2D(data)
				# segment data with sliding window 
				data, label	= segment_signal_without_transition(data, label, window_size)
				data		= data.reshape(int(data.shape[0]/window_size), window_size, 10, 11)
				# append new data and label
				data_inter	= np.vstack([data_inter, data])
				label_inter	= np.append(label_inter, label)
			else:
				pass
	
    # shuffle data
	index = np.array(range(0, len(label_inter)))
	np.random.shuffle(index)

	shuffled_data	= data_inter[index]
	shuffled_label 	= label_inter[index]

	return shuffled_data, shuffled_label
# set the initial parameters
dataset_dir		=	get_args().directory
window_size		=	get_args().window
begin_subject	=	get_args().begin
end_subject		=	get_args().end
output_dir		=	get_args().output_dir
set_store		=	True # get_args().set_store

# print out the initial parameters
print_top(dataset_dir, window_size, begin_subject, end_subject, output_dir, set_store)

# shuffle the data and labels
shuffled_data, shuffled_label = apply_mixup(dataset_dir, window_size, begin_subject, end_subject+1)

if (set_store == True):
	output_data = output_dir+"scott_data/"+str(begin_subject)+"_"+str(end_subject)+"_eeg_dataset.pkl"
	output_label= output_dir+"scott_data/"+str(begin_subject)+"_"+str(end_subject)+"_eeg_labels.pkl"
		
	with open(output_data, "wb") as fp:
		pickle.dump(shuffled_data, fp, protocol=4) 
	with open(output_label, "wb") as fp:
		pickle.dump(shuffled_label, fp)

# load the dataset and labels into separate files
with open(output_dir+"scott_data/"+str(begin_subject)+"_"+str(end_subject)+"_eeg_dataset.pkl", "rb") as fp:
  	datasets = pickle.load(fp)
with open(output_dir+"scott_data/"+str(begin_subject)+"_"+str(end_subject)+"_eeg_labels.pkl", "rb") as fp:
  	labels = pickle.load(fp)


# reshape the dataset to fit in a 2D mesh with time segments of 10 meshes
datasets = datasets.reshape(len(datasets), window_size, 10, 11, 1)
print(datasets.shape)


# apply the one hot naming method for the labels and then put them as an array
one_hot_labels = np.array(list(pd.get_dummies(labels)))
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
print(labels)

# produce a random array with 75% of values true
split = np.random.rand(len(datasets)) < 0.75

# use split to obtain 75% of data and labels
train_x = datasets[split]
train_y = labels[split]

train_sample = len(train_x)

# obtain the other 25% of the data and labels
test_x = datasets[~split] 
test_y = labels[~split]

test_sample = len(test_x)

# print a checkpoint with the time
print("**********("+time.asctime(time.localtime(time.time()))+") Load and Split dataset End **********\n")
print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions Begin: **********\n")


##########################################################################################
# Set model parameters
##########################################################################################

# Recurrent neural network parameters
n_lstm_layers = 2
n_fc_in = 1024
n_fc_out = 1024

# indicate how many classifications there will be
n_labels = 5

# define the size of the 2D mesh
input_height = 10
input_width = 11
input_channel_num = 1

# define pooling parameters
pooling_height 	= 2
pooling_width 	= 2
pooling_stride = 2

# define kernel parameters
kernel_height_1st	= 3
kernel_width_1st 	= 3

kernel_height_2nd	= 3
kernel_width_2nd 	= 3

kernel_height_3rd	= 3
kernel_width_3rd 	= 3

kernel_stride 	= 1
conv_channel_num = 32

# size of fully connected layer
fc_size = 1024

# introduce parameter for l2 regularisation (ridge regression)
lambda_loss_amount = 0.0005

# learning rate parameter for optimisation
learning_rate = 1e-4

# number of training epochs
training_epochs = 5

# define the size of a batch and therefore the number of batches per epoch
batch_size = 200
batch_num_per_epoch = train_x.shape[0]//batch_size

# defining batch parameters for accuracy training and testing
accuracy_batch_size = 300
train_accuracy_batch_num = train_x.shape[0]//accuracy_batch_size
test_accuracy_batch_num = test_x.shape[0]//accuracy_batch_size

X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, 1], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name = 'Y')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, kernel_stride):
  return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride):
	weight = weight_variable([filter_height, filter_width, in_channels, out_channels])
	bias = bias_variable([out_channels])
	return tf.nn.elu(tf.add(conv2d(x, weight, kernel_stride), bias))

def max_pooling(x, pooling_height, pooling_width, pooling_stride):
	return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size):
	fc_weight = weight_variable([x_size, fc_size])
	fc_bias = bias_variable([fc_size])
	return tf.nn.elu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size):
	readout_weight = weight_variable([x_size, readout_size])
	readout_bias = bias_variable([readout_size])
	return tf.add(tf.matmul(x, readout_weight), readout_bias)

print("**********("+time.asctime(time.localtime(time.time()))+") Define parameters and functions End **********\n")
print("**********("+time.asctime(time.localtime(time.time()))+") Define NN structure Begin: **********\n")

##########################################################################################
# Build Neural Network Architecture
##########################################################################################

# Apply first CNN layer
conv_1 = apply_conv2d(X, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)

# Apply second CNN layer
conv_2 = apply_conv2d(conv_1, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num*2, kernel_stride)

# Apply third CNN layer
conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num*2, conv_channel_num*4, kernel_stride)

shape = conv_3.get_shape().as_list()

pool_2_flat = tf.reshape(conv_3, [-1, shape[1]*shape[2]*shape[3]])
fc = apply_fully_connect(pool_2_flat, shape[1]*shape[2]*shape[3], fc_size)
fc = tf.reshape(fc, [300, 1024])
# dropout regularizer
# Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
fc_drop = tf.nn.dropout(fc, keep_prob)

""" # fc_drop size [batch_size*window_size, fc_size]
# lstm_in size [batch_size, window_size, fc_size]
lstm_in = tf.reshape(fc_drop, [-1, window_size, fc_size])	

###########################################################################################
# add lstm cell to network
###########################################################################################
# define lstm cell
cells = []
for _ in range(n_lstm_layers):
	cell = tf.contrib.rnn.BasicLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)
# cell = tf.contrib.rnn.LSTMBlockCell(n_fc_in, forget_bias=1.0)
# cell = tf.contrib.rnn.GRUBlockCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)
# cell = tf.contrib.rnn.GridLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)
# cell = tf.contrib.rnn.GLSTMCell(n_fc_in, forget_bias=1.0, state_is_tuple=True)
# cell = tf.contrib.rnn.GRUCell(n_fc_in, state_is_tuple=True)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
	cells.append(cell)
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

# output ==> [batch, step, n_fc_in]
output, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state, time_major=False)

# output ==> [step, batch, n_fc_in]
# output = tf.transpose(output, [1, 0, 2])

# only need the output of last time step
# rnn_output ==> [batch, n_fc_in]
# rnn_output = tf.gather(output, int(output.get_shape()[0])-1)
# print(type(rnn_output))
###################################################################
# another output method
output = tf.unstack(tf.transpose(output, [1, 0, 2]), name = 'lstm_out')
rnn_output = output[-1]
###################################################################

###########################################################################################
# fully connected and readout
###########################################################################################
# rnn_output ==> [batch, fc_size]
shape_rnn_out = rnn_output.get_shape().as_list()
# fc_out ==> [batch_size, n_fc_out]
fc_out = apply_fully_connect(rnn_output, shape_rnn_out[1], n_fc_out)
 
fc_drop = tf.nn.dropout(fc_out, keep_prob)"""
dropout_prob = 0.5
phase_train = tf.placeholder(tf.bool, name = 'phase_train')

# get the readouts
y_ = apply_readout(fc_drop, fc_size, n_labels)
print(y_)
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name = "y_pred")
y_posi = tf.nn.softmax(y_, name = "y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
	tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)
"""
# evaluation and training of the model using cross entropy cost and Adam Optimiser
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name = 'loss')
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

print("**********("+time.asctime(time.localtime(time.time()))+") Define NN structure End **********\n")
print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN Begin: **********\n")

##########################################################################################
# Run Neural Network Model
##########################################################################################

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	train_accuracy_save = np.zeros(shape=[0], dtype=float)
	test_accuracy_save 	= np.zeros(shape=[0], dtype=float)
	test_loss_save 		= np.zeros(shape=[0], dtype=float)
	train_loss_save 	= np.zeros(shape=[0], dtype=float)
	for epoch in range(training_epochs):
		cost_history = np.zeros(shape=[0], dtype=float)
		for b in range(batch_num_per_epoch):
			offset = (b * batch_size) % (train_y.shape[0] - batch_size) 
			batch_x = train_x[offset:(offset + batch_size), :, :, :, :]
			batch_x = batch_x.reshape(len(batch_x)*window_size, 10, 11, 1)
			batch_y = train_y[offset:(offset + batch_size), :]
			_, c = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob, phase_train: True})
			cost_history = np.append(cost_history, c)
		if(epoch % 1 == 0):
			train_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_accuracy	= np.zeros(shape=[0], dtype=float)
			test_loss 		= np.zeros(shape=[0], dtype=float)
			train_loss 		= np.zeros(shape=[0], dtype=float)
			for i in range(train_accuracy_batch_num):
				offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size) 
				train_batch_x = train_x[offset:(offset + accuracy_batch_size), :, :, :, :]
				train_batch_x = train_batch_x.reshape(len(train_batch_x)*window_size, 10, 11, 1)
				train_batch_y = train_y[offset:(offset + accuracy_batch_size), :]
				train_a, train_c = session.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0, phase_train: False})
				train_loss = np.append(train_loss, train_c)
				train_accuracy = np.append(train_accuracy, train_a)
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Training Cost: ", np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
			train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
			train_loss_save = np.append(train_loss_save, np.mean(train_loss))
			for j in range(test_accuracy_batch_num):
				offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
				test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
				test_batch_x = test_batch_x.reshape(len(test_batch_x)*window_size, 10, 11, 1)
				test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
				test_a, test_c = session.run([accuracy, cost], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, phase_train: False})
				test_accuracy = np.append(test_accuracy, test_a)
				test_loss = np.append(test_loss, test_c)
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Test Cost: ", np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy),"\n")
			test_accuracy_save 	= np.append(test_accuracy_save, np.mean(test_accuracy))
			test_loss_save 		= np.append(test_loss_save, np.mean(test_loss))
	test_accuracy 	= np.zeros(shape=[0], dtype=float)
	test_loss 		= np.zeros(shape=[0], dtype=float)
	test_pred		= np.zeros(shape=[0], dtype=float)
	test_true		= np.zeros(shape=[0, 5], dtype=float)
	test_posi		= np.zeros(shape=[0, 5], dtype=float)
	for k in range(test_accuracy_batch_num):
		offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
		test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
		test_batch_x = test_batch_x.reshape(len(test_batch_x)*window_size, 10, 11, 1)
		test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
		test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0, phase_train: False})
		test_t = test_batch_y
		test_accuracy 	= np.append(test_accuracy, test_a)
		test_loss 		= np.append(test_loss, test_c)
		test_pred 		= np.append(test_pred, test_p)
		test_true 		= np.vstack([test_true, test_t])
		test_posi		= np.vstack([test_posi, test_r])
	test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype = np.int8)
	test_true_list	= tf.argmax(test_true, 1).eval()

# recall
	test_recall = recall_score(test_true, test_pred_1_hot, average=None)
	# precision
	test_precision = precision_score(test_true, test_pred_1_hot, average=None)
	# f1 score
	test_f1 = f1_score(test_true, test_pred_1_hot, average=None)
	# auc
	test_auc = roc_auc_score(test_true, test_pred_1_hot, average=None)
	# confusion matrix
	confusion_matrix = confusion_matrix(test_true_list, test_pred)

	print("********************recall:", test_recall)
	print("*****************precision:", test_precision)
	print("******************test_auc:", test_auc)
	print("******************f1_score:", test_f1)
	print("*************roc_auc_score:", test_auc)
	print("**********confusion_matrix:\n", confusion_matrix)

	print("("+time.asctime(time.localtime(time.time()))+") Final Test Cost: ", np.mean(test_loss), "Final Test Accuracy: ", np.mean(test_accuracy))
	# save result
	# os.system("mkdir .\\result\\"+output_dir)
	result 	= pd.DataFrame({'epoch':range(1,epoch+2), "train_accuracy":train_accuracy_save, "test_accuracy":test_accuracy_save,"train_loss":train_loss_save,"test_loss":test_loss_save})
	ins 	= pd.DataFrame({'conv_1':conv_1_shape, 'pool_1':pool_1_shape, 'conv_2':conv_2_shape, 'pool_2':pool_2_shape, 'conv_3':conv_3_shape, 'pool_3':pool_3_shape, 'conv_4':conv_4_shape, 'pool_4':pool_4_shape, 'fc':fc_size,'accuracy':np.mean(test_accuracy), 'keep_prob': 1-dropout_prob,  'n_person':n_person, "calibration":calibration, 'sliding_window':window_size, "epoch":epoch+1, "norm":norm_type, "learning_rate":learning_rate, "regularization":regularization_method, "train_sample":train_sample, "test_sample":test_sample}, index=[0])
	summary = pd.DataFrame({'class':one_hot_labels, 'recall':test_recall, 'precision':test_precision, 'f1_score':test_f1, 'roc_auc':test_auc})

	writer = pd.ExcelWriter(output_dir+"/"+output_file+".xlsx") # pylint: disable=abstract-class-instantiated
	ins.to_excel(writer, 'condition', index=False)
	result.to_excel(writer, 'result', index=False)
	summary.to_excel(writer, 'summary', index=False)
	# fpr, tpr, auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	i = 0
	for key in one_hot_labels:
		fpr[key], tpr[key], _ = roc_curve(test_true[:, i], test_posi[:, i])
		roc_auc[key] = auc(fpr[key], tpr[key])
		roc = pd.DataFrame({"fpr":fpr[key], "tpr":tpr[key], "roc_auc":roc_auc[key]})
		roc.to_excel(writer, key, index=False)
		i += 1
	writer.save()
	with open(output_dir+"scott_data/confusion_matrix.pkl", "wb") as fp:
  		pickle.dump(confusion_matrix, fp)
	# save model
	saver = tf.train.Saver()
	saver.save(session, output_dir+"scott_data/model_"+output_file)

print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN End **********\n")"""