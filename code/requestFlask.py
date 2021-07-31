import requests
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import Counter

def mode(sample):
	c = Counter(sample)
	return [k for k, v in c.items() if v == c.most_common(1)[0][1]]

def mode2(sample):
	c = Counter(sample)
	value = []
	amount = []
	for entry in c.most_common(1):
		value.append(entry[0])
		amount.append(entry[1]/len(sample))
	return value, amount

def data_1Dto2D(data, Y=10, X=11):
	data_2D = np.zeros([Y, X])
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0,        0,        0,        0, 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0,        0,  data[0],        0, data[13],        0, 	   	 0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0,  data[1],        0,  data[2],        0,        0,        0, data[11],        0, data[12], 	 	 0) 
	data_2D[3] = (	  	 0,        0,  data[3],        0,        0,        0,        0,        0, data[10],        0, 		 0) 
	data_2D[4] = (       0,  data[4],        0,        0,        0,        0,        0,        0,        0,  data[9],        0) 
	data_2D[5] = ( 	   	 0, 	   0,  	   	 0, 	   0,        0,        0,        0, 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[6] = (	  	 0,  data[5],  	   	 0, 	   0,        0,        0,        0,        0,        0,  data[8], 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0,   	   0, 	     0,        0,        0, 	   0, 	     0, 	   0,        0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0,  data[6],        0,  data[7], 	   0, 	   	 0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0,        0, 		 0, 	   0, 	   	 0, 	   0, 		 0) 
	return data_2D

def dataset_1Dto2D(dataset_1D):
	dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		dataset_2D[i] = data_1Dto2D(dataset_1D[i])
	return dataset_2D

def norm_dataset(dataset_1D):
	norm_dataset_1D = np.zeros([dataset_1D.shape[0], 14])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
	return norm_dataset_1D

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data.nonzero()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def segment_signal_without_transition(data, window_size):
	for (start, end) in windows(data, window_size):
		if((len(data[start:end]) == window_size)):
			if(start == 0):
				segments = data[start:end]
			else:
				segments = np.vstack([segments, data[start:end]])
	return segments

def windows(data, size):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += (size/2)


# BASE = "http://127.0.0.1:5000/"

# response = requests.put(BASE + "1", {"action": 3})
# print(response.json())
# response = requests.get(BASE + "1")
# print(response.json())
# response = requests.put(BASE + "1", {"action": 10})
# print(response.json())
# response = requests.get(BASE + "1")
# print(response.json())

# Load meta graph and restore weights
sess=tf.Session()    
saver = tf.train.import_meta_graph('d:/ToRNeuralNetork/scott_real_model/model_10_108_convolutional_200_epoch.meta')
saver.restore(sess, tf.train.latest_checkpoint('d:/ToRNeuralNetork/scott_real_model/'))

# read in the recorded eeg set
eegData = pd.read_csv('d:/ToRNeuralNetork/Scott_EEG_Data/RealRightFist_EPOCPLUS_124895_2021.02.28T18.27.21+11.00.md.csv', usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], skiprows=1)
data = eegData.to_numpy()

# split the data (> 5000 represents the actual trial timing)
data_realfists = data

# normalise the data
data_realfists = norm_dataset(data_realfists)

# transform the signals into 2D format
data_realfists = dataset_1Dto2D(data_realfists)

# set the sample size to 10 (@128 Hz) and reshape the data to fit the model specs
window_size = 10
data = segment_signal_without_transition(data_realfists, window_size)
data = data.reshape(int(data.shape[0]/window_size), window_size, 10, 11)
data = data.reshape(len(data), window_size, 10, 11, 1)
data = data.reshape(len(data)*window_size, 10, 11, 1)
i = 0
data_buf = []


while i * 3000 + 3000 < data.shape[0]:
	data_buf.append(data[i*3000:i*3000+3000])
	i += 1

print(len(data_buf))

# load the variables from the saved model
graph = tf.get_default_graph()

# storing the eeg data
X = graph.get_tensor_by_name("X:0")

# stores the models prediction
y_pred = graph.get_tensor_by_name("y_pred:0")

# stores the dropout parameter
keep_prob = graph.get_tensor_by_name("keep_prob:0")

for sample in data_buf:
	# set the new x data to be the recorded eeg data
	feed_dict = { X: sample, keep_prob: 1 }

	# run the model given the new data
	pred = sess.run(y_pred, feed_dict)

	# split the predictions into a certain sample size
	pred_sample_size = 50
	prediction = np.split(pred, pred.size/pred_sample_size)

	# for each prediction set, calculate the mode
	fin_pred = []
	for arr in prediction:
		value, amount = mode2(arr)
		if (amount[0] > 0.7):
			fin_pred.append(value)

	# print out the model predictions
	print(fin_pred)