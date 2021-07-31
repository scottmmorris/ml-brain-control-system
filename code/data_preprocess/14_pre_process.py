#! /usr/bin/python3

########################################################
# EEG data preprocess for 3D
########################################################
import argparse
import os
import pyedflib
import numpy as np
import pandas as pd
import pickle
from scipy import stats

np.random.seed(0)
def get_args():
	parser = argparse.ArgumentParser()
	
	hpstr = "set dataset directory"
	parser.add_argument('-d', '--directory', default="d:/ToRNeuralNetork/Cascade-Parallel-master/raw_data/", nargs='*', type=str, help=hpstr)

	hpstr = "set window size"
	parser.add_argument('-w', '--window', default=10, nargs='*', type=int, help=hpstr)

	hpstr = "set begin person"
	parser.add_argument('-b', '--begin', default=1, nargs='?', type=int, help=hpstr)

	hpstr = "set end person"
	parser.add_argument('-e', '--end', default=108, nargs='?', type=int, help=hpstr)

	hpstr = "set output directory"
	parser.add_argument('-o', '--output_dir', default="d:/ToRNeuralNetork/Cascade-Parallel-master/", nargs='*', help=hpstr)

	hpstr = "set whether store data"
	parser.add_argument('--set_store', action='store_true', help=hpstr)

	args = parser.parse_args()
	return(args)
		   
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
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0,        0,        0,        0, 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0,        0, data[25],        0, data[27],        0, 	     0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0, data[29],            0, data[31],        0,        0,        0, data[35],        0,     data[37], 	 	 0) 
	data_2D[3] = (	  	 0,        0,      data[0],        0,        0,        0,        0,        0,  data[6],            0, 		 0) 
	data_2D[4] = (           0,        0,            0,        0,        0,        0,        0,        0,        0,            0,            0) 
	data_2D[5] = (	  	 0, data[44],            0,        0,        0,        0,        0,        0,        0,     data[45], 		 0) 
	data_2D[6] = (	  	 0,        0,            0,        0,        0,        0,        0,        0,        0,            0, 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0, data[55],        0,        0,        0, data[59], 	     0, 	   0, 		 0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0, data[60],        0, data[62], 	   0, 	     0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0,        0, 	 0, 	   0, 	     0, 	   0, 		 0) 
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
				labels = np.array(list(set(label[start:end])))
			else:
				segments = np.vstack([segments, data[start:end]])
				labels = np.append(labels, np.array(list(set(label[start:end]))))
				# labels = np.append(labels, stats.mode(label[start:end])[0][0])
	return segments, labels

def apply_mixup(dataset_dir, window_size, start=1, end=110):
	# initial empty label arrays
	label_inter	= np.empty([0])
	# initial empty data arrays
	data_inter	= np.empty([0, window_size, 10, 11])
	for j in range(start, end):
		if (j == 89):
			j = 109
		# get directory name for one subject
		data_dir = dataset_dir+"S"+format(j, '03d')
		# get task list for one subject
		task_list = [task for task in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, task))]
		for task in task_list:
			if(("R02" in task) or ("R05" in task) or ("R09" in task) or ("R13" in task)): # R02: eye closed; R05, R09, R13: motor imagery tasks
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
				if(data.shape[0]%10 != 0):
					j = data.shape[0]
					while(j%10!=0):
						j = j - 1
					data = data[0:j, :, :]
					label = label[0:j]
				shape = data.shape[0]
				data		= data.reshape(int(data.shape[0]/window_size), window_size, 10, 11)
				#label = label.reshape(int(shape/window_size), window_size)
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

if __name__ == '__main__':
	dataset_dir		=	get_args().directory
	window_size		=	get_args().window
	begin_subject	=	get_args().begin
	end_subject		=	get_args().end
	output_dir		=	get_args().output_dir
	set_store		=	True # get_args().set_store
	print_top(dataset_dir, window_size, begin_subject, end_subject, output_dir, set_store)

	shuffled_data, shuffled_label = apply_mixup(dataset_dir, window_size, begin_subject, end_subject+1)

	if (set_store == True):
		output_data = output_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_win_"+str(window_size)+".pkl"
		output_label= output_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_win_"+str(window_size)+".pkl"

		with open(output_data, "wb") as fp:
			pickle.dump(shuffled_data, fp, protocol=4) 
		with open(output_label, "wb") as fp:
			pickle.dump(shuffled_label, fp, protocol=4)
