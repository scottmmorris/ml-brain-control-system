#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations, read_events
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

import pickle

#print(__doc__)

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)

num_features = 4

label_train = np.empty([0, 36])
data_train = np.empty([0, 36, num_features])
label_test = np.empty([0, 9])
data_test = np.empty([0, 9, num_features])

begin_subject = 1
end_subject = 21
runs = [6, 10, 14]  # motor imagery: hands vs feet


for subject in range(begin_subject, end_subject):
	string = '____ Starting subject '+ str(subject) +' ____'
	print(string)

	raw_fnames = eegbci.load_data(subject, runs, path='d:/ToRNeuralNetork/Cascade-Parallel-master/MNEDataset/')
	raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
	eegbci.standardize(raw)  # set channel names
	montage = make_standard_montage('standard_1005')
	raw.set_montage(montage)

	# strip channel names of "." characters
	raw.rename_channels(lambda x: x.strip('.'))

	# Apply band-pass filter
	raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

	events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

	# Read epochs (train will be done only between 1 and 2s)
	# Testing will be done with a running classifier
	epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
	epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
	labels = epochs.events[:, -1] - 2

	# Define a monte-carlo cross-validation generator (reduce variance):
	epochs_data = epochs.get_data()
	epochs_data_train = epochs_train.get_data()
	cv = ShuffleSplit(10, test_size=0.2, random_state=42)
	cv_split = cv.split(epochs_data_train)

	# Assemble a classifier
	csp = CSP(n_components=num_features, reg=None, log=True, norm_trace=False)

	# plot CSP patterns estimated on full data for visualization
	csp.fit_transform(epochs_data, labels)

	#csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
	#plt.show()
	print(epochs_data.shape)
	print(epochs_data_train.shape)
	print(len(epochs_data))

	label_train_inter = np.empty([0, int((len(epochs_data) * 4 / 5))])
	data_train_inter = np.empty([0, int((len(epochs_data) * 4 / 5)), num_features])
	label_test_inter = np.empty([0, int((len(epochs_data) / 5))])
	data_test_inter = np.empty([0, int((len(epochs_data) / 5)), num_features])

	for train_idx, test_idx in cv_split:
		y_train, y_test = labels[train_idx], labels[test_idx]

		X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
		X_test = csp.transform(epochs_data_train[test_idx])
		print(X_train.shape)
		print(X_test.shape)
		X_train = X_train.reshape(1, int((len(epochs_data) * 4 / 5)), num_features)
		X_test = X_test.reshape(1, int((len(epochs_data) / 5)), num_features)
		y_train = y_train.reshape(1, int((len(epochs_data) * 4 / 5)))
		y_test = y_test.reshape(1, int((len(epochs_data) / 5)))
		data_train_inter	= np.append(data_train_inter, X_train, axis=0)
		label_train_inter	= np.append(label_train_inter, y_train, axis=0)
		data_test_inter	= np.append(data_test_inter, X_test, axis=0)
		label_test_inter	= np.append(label_test_inter, y_test,axis=0)

	data_train = np.vstack([data_train, data_train_inter])
	label_train = np.append(label_train, label_train_inter, axis=0)
	data_test	= np.vstack([data_test, data_test_inter])
	label_test	= np.append(label_test, label_test_inter, axis=0)


print(data_train.shape)
print(label_train.shape)
print(data_test.shape)
print(label_test.shape)

output_dir ="d:/ToRNeuralNetork/Cascade-Parallel-master/MNEDataset/Datafiles/"

output_data = output_dir+str(begin_subject)+"_"+str(end_subject)+"_data_train_"+str(num_features)+".pkl"
output_label= output_dir+str(begin_subject)+"_"+str(end_subject)+"_labels_train_"+str(num_features)+".pkl"
output_data_test = output_dir+str(begin_subject)+"_"+str(end_subject)+"_data_test_"+str(num_features)+".pkl"
output_label_test = output_dir+str(begin_subject)+"_"+str(end_subject)+"_labels_test_"+str(num_features)+".pkl"

with open(output_data, "wb") as fp:
	pickle.dump(data_train, fp, protocol=2) 
with open(output_label, "wb") as fp:
	pickle.dump(label_train, fp, protocol=2)
with open(output_data_test, "wb") as fp:
	pickle.dump(data_test, fp, protocol=2) 
with open(output_label_test, "wb") as fp:
	pickle.dump(label_test, fp, protocol=2)

