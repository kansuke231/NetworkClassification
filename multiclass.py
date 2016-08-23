#!/Users/ikeharakansuke/env/bin/python
from misc import *
from plot import plot_confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod):
	"""
	This functions returns a confusion matrix and NetworkTypeLabels
	"""

	if isSubType:
		NetworkTypeLabels = sorted(list(set(Y)), key=lambda sub_type: sub_to_main_type[sub_type])
	else:
		NetworkTypeLabels = sorted(list(set(Y))) # for NetworkType
	
	# type_to_int = dict([(e,i) for i,e in enumerate(NetworkTypeLabels)])
	# int_to_type = dict([(i,e) for i,e in enumerate(NetworkTypeLabels)])
	# Y = np.array([type_to_int[e] for e in Y])

	sss = StratifiedShuffleSplit(Y, 3, test_size=0.4, random_state=0)

	for train_index, test_index in sss:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]


	if samplingMethod == "RandomOver":
		random_over = RandomOverSampler()
		sampled_x, sampled_y = random_over.fit_sample(X_train, y_train)

	elif samplingMethod == "RandomUnder":
		random_under = RandomUnderSampler()
		sampled_x, sampled_y = random_under.fit_sample(X_train, y_train)
		from collections import Counter
		print Counter(sampled_y)

	elif samplingMethod == "SMOTE":
		sm = SMOTE(kind='regular', k=3)
		sm.fit(X_train, y_train)
		majority = sm.maj_c_
	
		all_X = []
		all_Y = []
	
		for network_type in NetworkTypeLabels:
			if network_type != majority:
				# extract elements of a pair of network types, i.e. the majority and one to be inflated
				X_extracted = np.concatenate((X_train[y_train == majority],X_train[y_train == network_type]),axis=0)
				Y_extracted = np.concatenate((y_train[y_train == majority],y_train[y_train == network_type]),axis=0)
				x_tmp, y_tmp = sm.fit_sample(X_extracted,Y_extracted)
				x = x_tmp[y_tmp == network_type]
				y = y_tmp[y_tmp == network_type]
				all_X.append(x)
				all_Y.append(y)
	
		all_X.append(X_train[y_train == majority])
		all_Y.append(y_train[y_train == majority])
	

		Xs = np.concatenate(tuple(all_X))
		Ys = np.concatenate(tuple(all_Y))
	
		sampled_x, sampled_y = sm.fit_sample(Xs,Ys)


	

	
	elif samplingMethod == "None":
		sampled_x, sampled_y = X_train, y_train

	random_forest = RandomForestClassifier()
	random_forest.fit(sampled_x,sampled_y)
	accuracy = random_forest.score(X_test,y_test)

	print "Feature Importance"
	print sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), feature_names), reverse=True)
	print "----------------------------------------------------"
	print "prediction: %f"%accuracy

	y_pred = random_forest.predict(X_test)
	cm = confusion_matrix(y_test, y_pred, labels=NetworkTypeLabels)
	return cm, NetworkTypeLabels,accuracy

def main():
	
	column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity",#"MeanGeodesicDistance",\
				    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	feature_names = ["ClusteringCoefficient","MeanGeodesicPath","Modularity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	isSubType = True
	at_least = 7
	X,Y,sub_to_main_type = init("features.csv", column_names, feature_names, isSubType, at_least)

	#X,Y = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType)
	#plot_scikit_lda_3d(X, Y)
	cm, NetworkTypeLabels,accuracy = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, "SMOTE")
	plot_confusion_matrix(cm, NetworkTypeLabels, sub_to_main_type, isSubType)



if __name__ == "__main__":
	main()


