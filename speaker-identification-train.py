# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from features import FeatureExtractor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import pickle


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

#data_dir = 'data/train' # directory where the data files are stored
data_dir = 'dummy data/train' # directory where the data files are stored

output_dir = 'training_output' # directory where the classifier(s) are stored

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

# the filenames should be in the form 'speaker-data-subject-1.csv', e.g. 'speaker-data-Erik-1.csv'.

#class_names = ["EDM", "Classical", "Metal"] # the set of classes, i.e. speakers
class_names = [] # the set of classes, i.e. speakers

data = np.zeros((0,8002)) #8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)

for filename in os.listdir(data_dir):
	if filename.endswith(".csv") and filename.startswith("speaker-data"):
		filename_components = filename.split("-") # split by the '-' character
		speaker = filename_components[2]
		print("Loading data for {}.".format(speaker))
		if speaker not in class_names:
			class_names.append(speaker)
		speaker_label = class_names.index(speaker)
		sys.stdout.flush()
		data_file = os.path.join(data_dir, filename)
		data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
		print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
		sys.stdout.flush()
		data = np.append(data, data_for_current_speaker, axis=0)

print("Found data for {} speakers : {}".format(len(class_names), ", ".join(class_names)))

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# Update this depending on how you compute your features
n_features = 1050

print("Extracting features and labels for {} audio windows...".format(data.shape[0]))
sys.stdout.flush()

X = np.zeros((0,n_features))
y = np.zeros(0,)

# change debug to True to show print statements we've included:
feature_extractor = FeatureExtractor(debug=False) 
nr_total_windows = 0
nr_bad_windows = 0
nr_windows_with_zeros = 0

for i,window_with_timestamp_and_label in enumerate(data):
	window = window_with_timestamp_and_label[1:-1]
	label = data[i,-1]
	nr_total_windows += 1
	try:
		x = feature_extractor.extract_features(window)
		if (len(x) != X.shape[1]):
			print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
		X = np.append(X, np.reshape(x, (1,-1)), axis=0)
		y = np.append(y, label)
	except:
		nr_bad_windows += 1
		if np.all((window == 0)):
			nr_windows_with_zeros += 1

print("{} windows found".format(nr_total_windows))
print("{} bad windows found, with {} windows with only zeros".format(nr_bad_windows, nr_windows_with_zeros))
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()


# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

print("\n")
print("---------------------- Decision Tree -------------------------")

total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0]

cv = KFold(n_splits=10, shuffle=True, random_state=None)
for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
	print("Fold {} : Training decision tree classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	tree.fit(X_train, y_train)
	print("Evaluating classifier over {} points...".format(len(y_test)))

	# predict the labels on the test data
	y_pred = tree.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2])
	print(conf)

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

print("Training decision tree classifier on entire dataset...")
tree.fit(X, y)

print("\n")
print("---------------------- Random Forest Classifier -------------------------")
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0]

for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print("Fold {} : Training Random Forest classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)

	print("Evaluating classifier over {} points...".format(len(y_test)))
	# predict the labels on the test data
	y_pred = clf.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2])
	print(conf)

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

# TODO: (optional) train other classifiers and print the average metrics using 10-fold cross-validation

print("\n")
print("---------------------- Naive Bayes Classifier -------------------------")
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0]

for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print("Fold {} : Training Naive Bayes classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	nb = GaussianNB()
	nb.fit(X_train, y_train)

	print("Evaluating classifier over {} points...".format(len(y_test)))
	# predict the labels on the test data
	y_pred = nb.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2])
	print(conf)

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  

print("\n")
print("---------------------- SVM Classifier -------------------------")
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0]

for i, (train_index, test_index) in enumerate(cv.split(X)):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	print("Fold {} : Training SVM classifier over {} points...".format(i, len(y_train)))
	sys.stdout.flush()
	clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
	clf.fit(X_train, y_train)

	print("Evaluating classifier over {} points...".format(len(y_test)))
	# predict the labels on the test data
	y_pred = clf.predict(X_test)

	# show the comparison between the predicted and ground-truth labels
	conf = confusion_matrix(y_test, y_pred, labels=[0,1,2])
	print(conf)

	accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
	precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
	recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

	total_accuracy += accuracy
	total_precision += precision
	total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  


# Set this to the best model you found, trained on all the data:
best_classifier = RandomForestClassifier(n_estimators=100)
best_classifier.fit(X,y) 

classifier_filename='classifier.pickle'
print("Saving best classifier to {}...".format(os.path.join(output_dir, classifier_filename)))
with open(os.path.join(output_dir, classifier_filename), 'wb') as f: # 'wb' stands for 'write bytes'
	pickle.dump(best_classifier, f)
# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

data_dir = 'dummy data/test' # directory where the data files are stored

output_dir = 'testing_output' # directory where the classifier(s) are stored

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

# the filenames should be in the form 'speaker-data-subject-1.csv', e.g. 'speaker-data-Erik-1.csv'.

class_names = [] # the set of classes, i.e. speakers

data = np.zeros((0,8002)) #8002 = 1 (timestamp) + 8000 (for 8kHz audio data) + 1 (label)

for filename in os.listdir(data_dir):
	if filename.endswith(".csv") and filename.startswith("speaker-data"):
		filename_components = filename.split("-") # split by the '-' character
		speaker = filename_components[2]
		print("Loading data for {}.".format(speaker))
		if speaker not in class_names:
			class_names.append(speaker)
		speaker_label = class_names.index(speaker)
		sys.stdout.flush()
		data_file = os.path.join(data_dir, filename)
		data_for_current_speaker = np.genfromtxt(data_file, delimiter=',')
		print("Loaded {} raw labelled audio data samples.".format(len(data_for_current_speaker)))
		sys.stdout.flush()
		data = np.append(data, data_for_current_speaker, axis=0)

print("Found data for {} speakers : {}".format(len(class_names), ", ".join(class_names)))
# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------
# Update this depending on how you compute your features
n_features = 1050

print("Extracting features and labels for {} audio windows...".format(data.shape[0]))
sys.stdout.flush()

X = np.zeros((0,n_features))
y = np.zeros(0,)

nr_total_windows = 0
nr_bad_windows = 0
nr_windows_with_zeros = 0

for i,window_with_timestamp_and_label in enumerate(data):
    window = window_with_timestamp_and_label[1:-1]
    label = data[i,-1]
    nr_total_windows += 1
    try:
        x = feature_extractor.extract_features(window)
        if (len(x) != X.shape[1]):
            print("Received feature vector of length {}. Expected feature vector of length {}.".format(len(x), X.shape[1]))
        X = np.append(X, np.reshape(x, (1,-1)), axis=0)
        y = np.append(y, label)
    except:
        nr_bad_windows += 1
        if np.all((window == 0)):
            nr_windows_with_zeros += 1

print("{} windows found".format(nr_total_windows))
print("{} bad windows found, with {} windows with only zeros".format(nr_bad_windows, nr_windows_with_zeros))

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()
# %%---------------------------------------------------------------------------
#
#		                  Predict on new data
#
# -----------------------------------------------------------------------------
total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0]

y_pred = best_classifier.predict(X)

# show the comparison between the predicted and ground-truth labels
conf = confusion_matrix(y, y_pred, labels=[0,1,2])
print(conf)

accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

print("The accuracy is {}".format(accuracy))  
print("The precision is {}".format(precision))    
print("The recall is {}".format(recall)) 
# %%
