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
import numpy as np
import scipy.linalg as la
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from keras.layers import Input, Dense
from keras.models import Model
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP

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
	parser.add_argument('-e', '--end', default=2, nargs='?', type=int, help=hpstr)

	hpstr = "set output directory"
	parser.add_argument('-o', '--output_dir', default="d:/ToRNeuralNetork/Cascade-Parallel-master/fe_datasets/", nargs='*', help=hpstr)

	hpstr = "set whether store data"
	parser.add_argument('--set_store', action='store_true', help=hpstr)

	hpstr = "set the feature extraction method"
	parser.add_argument('-f', '--feature', default='None', nargs='*', help=hpstr)

	args = parser.parse_args()
	return(args)
		   

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
	print(signal_labels)
	print(signal_labels.shape)
	f._close()
	del f
	return signal, signal_labels

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

def data_1Dto2D14(data, Y=10, X=11):
	data_2D = np.zeros([Y, X])
	data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0,        0,        0,        0, 	   0,  	     0, 	   0, 	 	 0) 
	data_2D[1] = (	  	 0, 	   0,  	   	 0,        0, data[25],        0, data[27],        0, 	   	 0,   	   0, 	 	 0) 
	data_2D[2] = (	  	 0, data[29],        0, data[31],        0,        0,        0, data[35],        0, data[37], 	 	 0) 
	data_2D[3] = (	  	 0,        0,  data[0],        0,        0,        0,        0,        0,  data[6],        0, 		 0) 
	data_2D[4] = (       0,        0,        0,        0,        0,        0,        0,        0,        0,        0,        0) 
	data_2D[5] = (	  	 0, data[44],        0,        0,        0,        0,        0,        0,        0, data[45], 		 0) 
	data_2D[6] = (	  	 0,        0,        0,        0,        0,        0,        0,        0,        0,        0, 		 0) 
	data_2D[7] = (	  	 0, 	   0, 	 	 0, data[55],        0,        0,        0, data[59], 	   	 0, 	   0, 		 0) 
	data_2D[8] = (	  	 0, 	   0, 	 	 0, 	   0, data[60],        0, data[62], 	   0, 	   	 0, 	   0, 		 0) 
	data_2D[9] = (	  	 0, 	   0, 	 	 0, 	   0, 	     0,        0, 		 0, 	   0, 	   	 0, 	   0, 		 0) 
	return data_2D

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

def dataset_1Dto2D(dataset_1D):
	dataset_2D = np.zeros([dataset_1D.shape[0], 10, 11])
	for i in range(dataset_1D.shape[0]):
		dataset_2D[i] = data_1Dto2D(dataset_1D[i])
	return dataset_2D

def apply_mixup(dataset_dir, window_size, start=1, end=110, feature='None'):
	# initial empty label arrays
	label_inter = np.empty([0])
	# initial empty data arrays
	data_inter = np.empty([0, 64])
	#data_inter	= np.empty([0, 10, 11])
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
				#data		= dataset_1Dto2D(data)
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
	X = shuffled_data

	if(feature=='PCA'):
		# use pca feature extraction
		pca = PCA(n_components=32)
		X = pca.fit_transform(shuffled_data)
	elif(feature=='ICA'):
		# use ica feature extraction
		ica = FastICA(n_components=32)
		X = ica.fit_transform(shuffled_data)
	elif(feature=='LDA'):
		# use lda feature extraction
		lda = LinearDiscriminantAnalysis(n_components=14)
		X = lda.fit(shuffled_data, shuffled_label).transform(shuffled_data)
	elif(feature=='LLE'):
		# use lle feature extraction
		embedding = LocallyLinearEmbedding(n_components=14)
		X = embedding.fit_transform(shuffled_data)
	elif(feature=='SNE'):
		# use t-SNE feature extraction
		start = time.process_time()
		tsne = TSNE(n_components=14, verbose=1, perplexity=40, n_iter=300)
		X = tsne.fit_transform(shuffled_data)
		print(time.process_time() - start)
	elif(feature=='AUE'):
		# use autoencoder feature extraction
		input_layer = Input(shape=(shuffled_data.shape[1],))
		encoded = Dense(32, activation='relu')(input_layer)
		decoded = Dense(shuffled_data.shape[1], activation='softmax')(encoded)
		autoencoder = Model(input_layer, decoded)
		autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

		X1, X2, Y1, Y2 = train_test_split(shuffled_data, shuffled_data, test_size=0.3, random_state=101)

		autoencoder.fit(X1, Y1, epochs=100, batch_size=300, shuffle=True, verbose = 30, validation_data=(X2, Y2))

		encoder = Model(input_layer, encoded)
		X = encoder.predict(shuffled_data)
	elif(feature=='CSP'):
		#ch_names = list(shuffled_data.columns[1:])
		data = np.concatenate((np.array(shuffled_data),shuffled_label))  
		raw = RawArray(data,info,verbose=False)
		events = find_events(raw,stim_channel='Replace', verbose=False)
		picks = pick_types(raw.info,eeg=True)
		epochs = Epochs(raw, events, {'during' : 1}, -2, -0.5, proj=False, picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=False)
		info = mne.create_info(ch_names=['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9' , 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'], sfreq=160, ch_types='eeg')
		csp = CSP(n_components=32, reg=None, log=True, norm_trace=False)
		csp.fit_transform(shuffled_data, shuffled_label)
		csp.plot_patterns(info, ch_type='eeg', units='Patterns (AU)', size=1.5)
	else:
		pass
	
	"""X_df = pd.DataFrame(data = X, columns = ['PC1', 'PC2', 'PC3']) #, 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14'])
	df = pd.DataFrame(data = shuffled_label, columns = ['class'])
	X_df = pd.concat([X_df, df['class']], axis = 1)
	X_df['class'] = LabelEncoder().fit_transform(X_df['class'])
	X_df.head()

	print('------ Plotting Feature Correlation -------')

	fig = figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(111, projection='3d')

	classes = [4, 3, 2, 1, 0]
	colors = ['r', 'b', 'g', 'y', 'c']
	for clas, color in zip(classes, colors):
		ax.scatter(X_df.loc[X_df['class'] == clas, 'PC1'], X_df.loc[X_df['class'] == clas, 'PC2'], X_df.loc[X_df['class'] == clas, 'PC3'], c = color)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.legend(['Close', 'Right', 'Left', 'Both', 'Feet'])
	ax.grid()
	plt.show()"""
	print(X.shape)
	return X, shuffled_label

if __name__ == '__main__':
	dataset_dir		=	get_args().directory
	window_size		=	get_args().window
	begin_subject	=	get_args().begin
	end_subject		=	get_args().end
	output_dir		=	get_args().output_dir
	set_store		=	True # get_args().set_store
	feature			=	'CSP'

	shuffled_data, shuffled_label = apply_mixup(dataset_dir, window_size, begin_subject, end_subject+1, feature)

	if (set_store == True):
		output_data = output_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_dataset_com_32_"+feature+".pkl"
		output_label= output_dir+str(begin_subject)+"_"+str(end_subject)+"_shuffle_labels_com_32_"+feature+".pkl"

		with open(output_data, "wb") as fp:
			pickle.dump(shuffled_data, fp, protocol=2) 
		with open(output_label, "wb") as fp:
			pickle.dump(shuffled_label, fp, protocol=2)

	""" print(label_inter.shape)
	print(data_inter.shape)
	eye_close = np.where(label_inter == 'eye_close')
	labels_1 = label_inter[eye_close]
	data_1 = data_inter[eye_close]
	right_fist = np.where(label_inter == 'image_open&close_right_fist')
	labels_2 = label_inter[right_fist]
	data_2 = data_inter[right_fist]
	left_fist = np.where(label_inter == 'image_open&close_left_fist')
	labels_3 = label_inter[left_fist]
	data_3 = data_inter[left_fist]
	both_feet = np.where(label_inter == 'image_open&close_both_feet')
	labels_4 = label_inter[both_feet]
	data_4 = data_inter[both_feet]
	both_fists = np.where(label_inter == 'image_open&close_both_fists')
	labels_5 = label_inter[both_fists]
	data_5 = data_inter[both_fists]
	filters = mne.decoding.CSP()
	filters.fit(data_inter, label_inter) """