#! /usr/bin/python3
###########################################################################
# implement 3D cnn for EEG decode
###########################################################################
from cnn_class import cnn
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

np.random.seed(33)

###########################################################################
# set model parameters
###########################################################################
# kernel parameter
kernel_depth_1st	= 3
kernel_height_1st	= 3
kernel_width_1st 	= 3

kernel_depth_2nd	= 3
kernel_height_2nd	= 3
kernel_width_2nd 	= 3

kernel_depth_3rd	= 3
kernel_height_3rd	= 3
kernel_width_3rd 	= 3

kernel_stride		= 1

conv_channel_num	= 32

# pooling parameter
pooling_depth_1st 	= "None"
pooling_height_1st 	= "None"
pooling_width_1st 	= "None"

pooling_depth_2nd 	= "None"
pooling_height_2nd 	= "None"
pooling_width_2nd	= "None"

pooling_depth_3rd 	= "None"
pooling_height_3rd 	= "None"
pooling_width_3rd 	= "None"

pooling_stride		= "None"

# full connected parameter
fc_size 			= 1024


###########################################################################
# set dataset parameters
###########################################################################
# input channel
input_channel_num = 1

# window size
window_size = 10

# input depth
input_depth = window_size

# input height 
input_height = 10

# input width
input_width = 11

# prediction class
num_labels = 5

# train test split
train_test_split = 0.75

# id of begin subject
begin_subject = 1

# id of end subject
end_subject = 20

# dataset directory
dataset_dir = "d:/ToRNeuralNetork/Cascade-Parallel-master/3D_CNN/raw_data/"

# load dataset and label
with open(dataset_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_3D_win_"+str(window_size)+".pkl", "rb") as fp:
  	datasets = pickle.load(fp)
with open(dataset_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_3D_win_"+str(window_size)+".pkl", "rb") as fp:
  	labels = pickle.load(fp)

# reshape dataset
datasets = datasets.reshape(len(datasets), window_size, 10, 11, 1)

# set label to one hot
one_hot_labels = np.array(list(pd.get_dummies(labels)))
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)

split = np.random.rand(len(datasets)) < train_test_split

train_x = datasets[split] 
train_y = labels[split]

test_x = datasets[~split] 
test_y = labels[~split]

# print label
print(one_hot_labels)

###########################################################################
# set training parameters
###########################################################################
# set learning rate
learning_rate = 1e-4

# set maximum traing epochs
training_epochs = 20

# set batch size
batch_size = 100

# set dropout probability
dropout_prob = 0.5

# set whether use L2 regularization
enable_penalty = True

# set L2 penalty
lambda_loss_amount = 0.0005

# set train batch number per epoch
batch_num_per_epoch = train_x.shape[0]//batch_size

# set test batch number per epoch
accuracy_batch_size = 300
train_accuracy_batch_num = train_x.shape[0]//accuracy_batch_size
test_accuracy_batch_num = test_x.shape[0]//accuracy_batch_size

###########################################################################
# for output record
###########################################################################

# shape of cnn layer
conv_1_shape = str(kernel_depth_1st)+"*"+str(kernel_height_1st)+"*"+str(kernel_width_1st)+"*"+str(kernel_stride)+"*"+str(conv_channel_num)
pool_1_shape = str(pooling_height_1st)+"*"+str(pooling_width_1st)+"*"+str(pooling_stride)+"*"+str(conv_channel_num)

conv_2_shape = str(kernel_depth_2nd)+"*"+str(kernel_height_2nd)+"*"+str(kernel_width_2nd)+"*"+str(kernel_stride)+"*"+str(conv_channel_num*2)
pool_2_shape = str(pooling_depth_2nd)+"*"+str(pooling_height_2nd)+"*"+str(pooling_width_2nd)+"*"+str(pooling_stride)+"*"+str(conv_channel_num*2)

conv_3_shape = str(kernel_depth_3rd)+"*"+str(kernel_height_3rd)+"*"+str(kernel_width_3rd)+"*"+str(kernel_stride)+"*"+str(conv_channel_num*4)
pool_3_shape = str(pooling_depth_3rd)+"*"+str(pooling_height_3rd)+"*"+str(pooling_width_3rd)+"*"+str(pooling_stride)+"*"+str(conv_channel_num*4)

# regularization method
if enable_penalty:
	regularization_method = 'dropout+l2'
else:
	regularization_method = 'dropout'

# result output
result_dir = "d:/ToRNeuralNetork/Cascade-Parallel-master/"
result_dir2 = "d:\\ToRNeuralNetork\\Cascade-Parallel-master\\"
output_dir 	= "3d_conv_3l_"+str(begin_subject)+"_"+str(end_subject)+"_fc_"+str(fc_size)+"_"+regularization_method+"_"+str(train_test_split*100)
output_file = "3d_conv_3l_"+str(begin_subject)+"_"+str(end_subject)+"_fc_"+str(fc_size)+"_"+regularization_method+"_"+str(train_test_split*100)

os.system("mkdir "+result_dir2+output_dir)
###########################################################################
# build network
###########################################################################

# instance cnn class
cnn_3d = cnn()

# input placeholder
X = tf.placeholder(tf.float32, shape=[None, input_depth, input_height, input_width, input_channel_num], name = 'X')
Y = tf.placeholder(tf.float32, shape=[None, num_labels], name = 'Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# first CNN layer
conv_1 = cnn_3d.apply_conv3d(X, kernel_depth_1st, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride)
# pool_1 = cnn_3d.apply_max_pooling3d(conv_1, pooling_depth, pooling_height, pooling_width, pooling_stride)

# second CNN layer
conv_2 = cnn_3d.apply_conv3d(conv_1, kernel_depth_2nd, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num*2, kernel_stride)
# pool_2 = cnn_3d.apply_max_pooling3d(conv_2, pooling_depth, pooling_height, pooling_width, pooling_stride)

# third CNN layer
conv_3 = cnn_3d.apply_conv3d(conv_2, kernel_depth_3rd, kernel_height_3rd, kernel_width_3rd, conv_channel_num*2, conv_channel_num*4, kernel_stride)
# pool_3 = cnn_3d.apply_max_pooling3d(conv_3, pooling_depth, pooling_height, pooling_width, pooling_stride)


# flattern the last layer of cnn
shape = conv_3.get_shape().as_list()
conv_3_flat = tf.reshape(conv_3, [-1, shape[1]*shape[2]*shape[3]*shape[4]])

# fully connected layer
fc = cnn_3d.apply_fully_connect(conv_3_flat, shape[1]*shape[2]*shape[3]*shape[4], fc_size)

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
fc_drop = tf.nn.dropout(fc, keep_prob)	

# readout layer
y_ = cnn_3d.apply_readout(fc_drop, fc_size, num_labels)

# probability prediction 
y_posi = tf.nn.softmax(y_, name = "y_posi")

# class prediction 
y_pred = tf.argmax(y_posi, 1, name = "y_pred")

# l2 regularization
l2 = lambda_loss_amount * sum(
	tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
	# cross entropy cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y)) + l2
else:
	# cross entropy cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')

# set training SGD optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))

# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

###########################################################################
# train test and save result
###########################################################################

# run with gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
	session.run(tf.global_variables_initializer())
	train_accuracy_save = np.zeros(shape=[0], dtype=float)
	test_accuracy_save 	= np.zeros(shape=[0], dtype=float)
	test_loss_save 		= np.zeros(shape=[0], dtype=float)
	train_loss_save 	= np.zeros(shape=[0], dtype=float)
	for epoch in range(training_epochs):
		cost_history = np.zeros(shape=[0], dtype=float)
		# training process
		for b in range(batch_num_per_epoch):
			offset = (b * batch_size) % (train_y.shape[0] - batch_size) 
			batch_x = train_x[offset:(offset + batch_size), :, :, :, :]
			batch_y = train_y[offset:(offset + batch_size), :]
			_, c = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1-dropout_prob})
			cost_history = np.append(cost_history, c)
		# calculate train and test accuracy after each training epoch
		if(epoch%1 == 0):
			train_accuracy 	= np.zeros(shape=[0], dtype=float)
			test_accuracy	= np.zeros(shape=[0], dtype=float)
			test_loss 		= np.zeros(shape=[0], dtype=float)
			train_loss 		= np.zeros(shape=[0], dtype=float)
			for i in range(train_accuracy_batch_num):
				offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size) 
				train_batch_x = train_x[offset:(offset + accuracy_batch_size), :, :, :, :]
				train_batch_y = train_y[offset:(offset + accuracy_batch_size), :]
				
				train_a, train_c = session.run([accuracy, cost], feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0})
				
				train_loss = np.append(train_loss, train_c)
				train_accuracy = np.append(train_accuracy, train_a)
			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Training Cost: ", np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
			train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
			train_loss_save = np.append(train_loss_save, np.mean(train_loss))
			# calculate test accuracy after each training epoch
			for j in range(test_accuracy_batch_num):
				offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
				test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
				test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
				
				test_a, test_c = session.run([accuracy, cost], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0})
				
				test_accuracy = np.append(test_accuracy, test_a)
				test_loss = np.append(test_loss, test_c)

			print("("+time.asctime(time.localtime(time.time()))+") Epoch: ", epoch+1, " Test Cost: ", np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy),"\n")
			test_accuracy_save 	= np.append(test_accuracy_save, np.mean(test_accuracy))
			test_loss_save 		= np.append(test_loss_save, np.mean(test_loss))
###########################################################################
# save result and model after training 
###########################################################################
	test_accuracy 	= np.zeros(shape=[0], dtype=float)
	test_loss 		= np.zeros(shape=[0], dtype=float)
	test_pred		= np.zeros(shape=[0], dtype=float)
	test_true		= np.zeros(shape=[0, 5], dtype=float)
	test_posi		= np.zeros(shape=[0, 5], dtype=float)
	for k in range(test_accuracy_batch_num):
		offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size) 
		test_batch_x = test_x[offset:(offset + accuracy_batch_size), :, :, :, :]
		test_batch_y = test_y[offset:(offset + accuracy_batch_size), :]
		
		test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi], feed_dict={X: test_batch_x, Y: test_batch_y, keep_prob: 1.0})
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
	# roc auc score
	test_auc = roc_auc_score(test_true, test_pred_1_hot, average=None)
	# confusion matrix
	confusion_matrix = confusion_matrix(test_true_list, test_pred)

	with open(result_dir+output_dir+"/confusion_matrix.pkl", "wb") as fp:
  		pickle.dump(confusion_matrix, fp)

	print("("+time.asctime(time.localtime(time.time()))+") Final Test Cost: ", np.mean(test_loss), "Final Test Accuracy: ", np.mean(test_accuracy))
	# save result
	result 	= pd.DataFrame({'epoch':range(1,epoch+2), "train_accuracy":train_accuracy_save, "test_accuracy":test_accuracy_save,"train_loss":train_loss_save,"test_loss":test_loss_save})
	ins 	= pd.DataFrame({'conv_1':conv_1_shape, 'pool_1':pool_1_shape, 'conv_2':conv_2_shape, 'pool_2':pool_2_shape, 'conv_3':conv_3_shape, 'pool_3':pool_3_shape, "begin_subject":begin_subject, "end_subject":end_subject, 'fc':fc_size,'accuracy':np.mean(test_accuracy), 'keep_prob': 1-dropout_prob, 'sliding_window':window_size, "epoch":epoch+1, "learning_rate":learning_rate, "regularization":regularization_method}, index=[0])
	summary = pd.DataFrame({'class':one_hot_labels, 'recall':test_recall, 'precision':test_precision, 'f1_score':test_f1, 'roc_auc':test_auc})

	writer = pd.ExcelWriter(result_dir+output_dir+"/"+output_file+"0002"1.xlsx") # pylint: disable=abstract-class-instantiated
	# save model implementation paralmeters
	ins.to_excel(writer, 'condition', index=False)
	# save train/test accuracy and loss for each epoch
	result.to_excel(writer, 'result', index=False)
	# save recall/precision/f1 for each class
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

	# save model
	saver = tf.train.Saver()
	saver.save(session, result_dir+output_dir+"/model_"+output_file)
print("**********("+time.asctime(time.localtime(time.time()))+") Train and Test NN End **********\n")































