#!/Users/ikeharakansuke/env/bin/python
from __future__ import division
from misc import init
import numpy as np
from multiclass import multiclass_classification
from plot import plot_confusion_matrix
from LDA import LinearDiscriminantAnalysis


def sum_confusion_matrix(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod, N):
	accum_matrix, NetworkTypeLabels, accum_acc = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod)
	for i in range(N):
		cm, _, accuracy = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod)
		accum_matrix += cm
		accum_acc += accuracy
	print X[0]
	return accum_matrix, NetworkTypeLabels, accum_acc



def main():
	column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity","DegreeAssortativity",#"MeanGeodesicDistance",\
				    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	feature_names = ["ClusteringCoefficient","Modularity","DegreeAssortativity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"] #"MeanGeodesicDistance"
	isSubType = True
	at_least = 6
	X,Y,sub_to_main_type = init("features.csv", column_names, feature_names, isSubType, at_least)
	N = 100
	
	Matrix_smote, NetworkTypeLabels, sum_accuracy = sum_confusion_matrix(X, Y, sub_to_main_type, feature_names, isSubType, "RandomUnder", N)
	average_matrix = np.array(map(lambda row: map(lambda e: e/N ,row), Matrix_smote))
	print "average accuracy: %f"%(float(sum_accuracy)/float(N))
	plot_confusion_matrix(average_matrix, NetworkTypeLabels, sub_to_main_type, isSubType)
	
	
	# for i in range(10):
	# 	cm, NetworkTypeLabels = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, "SMOTE")
		
if __name__ == '__main__':
	main()