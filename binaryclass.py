#!/Users/ikeharakansuke/env/bin/python
from misc import *
from plot import plot_distance_matrix, MDS_plot
import numpy as np
import sys
from itertools import permutations
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from unbalanced_dataset.over_sampling import SMOTE


def AUC(features, Y):
	"""
	Returns an AUC score for a given pair of classes.
	"""
	v = DictVectorizer(sparse=False)
	X = v.fit_transform(features)
	Y = np.array(Y)
	print "positive: %d negative: %d"%(np.count_nonzero(Y),len(Y)-np.count_nonzero(Y))
	ratio = float(np.count_nonzero(Y==0))/float(np.count_nonzero(Y==1))	
	
	#if ratio < 1.0:
	#	ratio = 0

	

	sss = StratifiedShuffleSplit(Y, 3, test_size=0.3, random_state=0)
	for train_index, test_index in sss:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]


	# Over-sampling with SMOTE
	#print "y_train", y_train
	smote = SMOTE(ratio=ratio, verbose=False, kind='regular', k=3)
	smox, smoy = smote.fit_transform(X_train, y_train)



	random_forest = RandomForestClassifier()
	random_forest.fit(smox, smoy)
	#random_forest.fit(X_train, y_train)


	y_pred = random_forest.predict(X_test)
	#print "y_pred:",y_pred
	#print "y_test:",y_test
	return roc_auc_score(y_test, y_pred)

def filter_pair(features, labels, pair):
	"""
	Filters out the features and labels for a given pair of classes.
	labels are conveted into either 1(positive) or 0(negative).
	"""
	f_filtered = []
	l_filtered = []
	positive, negative = pair
	for feature, label in zip(features,labels):
		if label == positive:
			f_filtered.append(feature)
			l_filtered.append(1)
		elif label == negative:
			f_filtered.append(feature)
			l_filtered.append(0)

	return f_filtered, l_filtered



def binary_classification(features, labels, feature_names, isSubType):

	sub_to_main_type = dict((SubType,NetworkType) for gml, NetworkType, SubType in labels)
	
	if isSubType:

		Y_temp1 = [SubType for gml, NetworkType, SubType in labels]
		Y_temp2 = [(i,y) for i,y in enumerate(Y_temp1) if Y_temp1.count(y) >= 6]
		Y = map(lambda (i,y): y, Y_temp2)
		indices = map(lambda (i,y): i, Y_temp2)
		features_temp = [features[i] for i in indices]
		features = features_temp

		NetworkTypeLabels = sorted(list(set(Y)), key=lambda sub_type: sub_to_main_type[sub_type])
	else:
		Y = [NetworkType for gml, NetworkType, SubType in labels]
		NetworkTypeLabels = sorted(list(set(Y))) # for NetworkType

	category_to_index = lambda category: NetworkTypeLabels.index(category)
	distance_matrix = [[0 for i in range(len(NetworkTypeLabels))] for j in range(len(NetworkTypeLabels))]

	pairs = permutations(set(Y),r=2)
	for pair in pairs:
		print "------------------------------------------------------"
		print pair
		fs,ls = filter_pair(features, Y, pair)
		print ls
		if len([1 for l in ls if l == 1]) > len([1 for l in ls if l == 0]):
			auc_sigma = []
			for i in range(50):
				auc_sigma.append(AUC(fs, ls))
			#auc_score = AUC(fs, ls)
			print auc_sigma
			auc_score = np.average(auc_sigma)
			print "AUC: %f"%auc_score
			i = category_to_index(pair[0])
			j = category_to_index(pair[1])
			distance_matrix[i][j] = abs(auc_score - 0.5) 
			distance_matrix[j][i] = abs(auc_score - 0.5) 

	return np.array(distance_matrix), NetworkTypeLabels


def main():
		
	column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity",#"MeanGeodesicDistance",\
				    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	feature_names = ["ClusteringCoefficient","MeanGeodesicPath","Modularity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	isSubType = True
	at_least = 6
	#X,Y,sub_to_main_type = init("features.csv", column_names, feature_names, isSubType, at_least)
	network_dict = data_read("features.csv", *column_names)
	network_dict = filter_float(network_dict)
	features,labels = XY_generator(network_dict)
	
	sub_to_main_type = dict((SubType,NetworkType) for gml, NetworkType, SubType in labels)
	
	distance_matrix, NetworkTypeLabels = binary_classification(features, labels, feature_names, isSubType=isSubType)
	plot_distance_matrix(distance_matrix, NetworkTypeLabels, sub_to_main_type, isSubType=isSubType)
	MDS_plot(distance_matrix, NetworkTypeLabels)


	



if __name__ == "__main__":
	main()


