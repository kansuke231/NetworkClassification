from misc import *
from plot import plot_confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.over_sampling import RandomOverSampler


def multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod):
	"""
	This functions returns a confusion matrix and NetworkTypeLabels
	"""

	if isSubType:
		NetworkTypeLabels = sorted(list(set(Y)), key=lambda sub_type: sub_to_main_type[sub_type])
	else:
		NetworkTypeLabels = sorted(list(set(Y))) # for NetworkType
	
	sss = StratifiedShuffleSplit(Y, 3, test_size=0.4, random_state=0)

	for train_index, test_index in sss:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]


	if samplingMethod == "Random":
		random_over = RandomOverSampler()
		sampled_x, sampled_y = random_over.fit_transform(X_train, y_train)

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
				x_tmp, y_tmp = sm.fit_transform(X_extracted,Y_extracted)
				x = x_tmp[y_tmp == network_type]
				y = y_tmp[y_tmp == network_type]
				print x.shape, y.shape
				all_X.append(x)
				all_Y.append(y)
	
		all_X.append(X_train[y_train == majority])
		all_Y.append(y_train[y_train == majority])
	

		Xs = np.concatenate(tuple(all_X))
		Ys = np.concatenate(tuple(all_Y))
	
		sampled_x, sampled_y = sm.fit_transform(Xs,Ys)
	
	elif samplingMethod == "None":
		sampled_x, sampled_y = X_train, y_train

	random_forest = RandomForestClassifier()
	random_forest.fit(sampled_x,sampled_y)

	print "Feature Importance"
	print sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), feature_names), reverse=True)
	print "----------------------------------------------------"
	print "prediction: %f"%random_forest.score(X_test,y_test)

	y_pred = random_forest.predict(X_test)
	cm = confusion_matrix(y_test, y_pred, labels=NetworkTypeLabels)
	return cm, NetworkTypeLabels

def main():
	
	column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity",#"MeanGeodesicDistance",\
				    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	feature_names = ["ClusteringCoefficient","MeanGeodesicPath","Modularity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	isSubType = True
	at_least = 6
	X,Y,sub_to_main_type = init("features.csv", column_names, feature_names, isSubType, at_least)

	#X,Y = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType)
	#plot_scikit_lda_3d(X, Y)
	cm, NetworkTypeLabels = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, "SMOTE")
	plot_confusion_matrix(cm, NetworkTypeLabels, sub_to_main_type, isSubType)

	



if __name__ == "__main__":
	main()


