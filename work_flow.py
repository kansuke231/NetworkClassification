#!/Users/ikeharakansuke/env/bin/python
from __future__ import division
from misc import init
import numpy as np
from multiclass import multiclass_classification
from plot import plot_confusion_matrix
from LDA import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx

def plot_feature_importance(Ls, feature_names):
	Ls = map(lambda x: map(lambda y: y[1],x), Ls)
	Ls = zip(*Ls) # Ls = [("ClustersingCoefficient", "ClustersingCoefficient", ..),]
	
	freq = {f: [] for f in feature_names}

	for freq_fs in Ls:
		for f in feature_names:
			freq[f].append(freq_fs.count(f))

	
	jet = plt.get_cmap('jet')
	cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(freq))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

	iterate = list(enumerate(freq.keys()))
	first = iterate[0]
	colorVal = scalarMap.to_rgba(first[0])
	p = plt.bar(range(len(feature_names)), freq[first[1]],  0.35, color=colorVal)
	prev = freq[first[1]] # previous stack

	ps = [p] # storing axis objects

	for i,k in iterate[1:]:
		colorVal = scalarMap.to_rgba(i)
		p = plt.bar(range(len(feature_names)), freq[k],  0.35, color=colorVal, bottom=prev)
		prev = map(lambda x: x[0]+x[1] , zip(prev,freq[k]))
		ps.append(p)

	plt.legend(ps,freq.keys(), bbox_to_anchor=(1.1, 0.4),prop={'size':12})
	plt.show()

def sum_confusion_matrix(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod, N):
	accum_matrix, NetworkTypeLabels, accum_acc, feature_importances = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod)
	list_important_features = [feature_importances]
	for i in range(N - 1):
		cm, _, accuracy, feature_importances = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, samplingMethod)
		accum_matrix += cm
		accum_acc += accuracy
		list_important_features.append(feature_importances)
	return accum_matrix, NetworkTypeLabels, accum_acc, list_important_features



def main():
	column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity","DegreeAssortativity",#"MeanGeodesicDistance",\
				    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	feature_names = ["ClusteringCoefficient","Modularity","DegreeAssortativity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"] #"MeanGeodesicDistance"
	isSubType = False
	at_least = 6
	X,Y,sub_to_main_type = init("features.csv", column_names, feature_names, isSubType, at_least)
	N = 100
	
	Matrix, NetworkTypeLabels, sum_accuracy, list_important_features = sum_confusion_matrix(X, Y, sub_to_main_type, feature_names, isSubType, "None", N)
	plot_feature_importance(list_important_features, feature_names)
	average_matrix = np.array(map(lambda row: map(lambda e: e/N ,row), Matrix))
	print "average accuracy: %f"%(float(sum_accuracy)/float(N))
	plot_confusion_matrix(average_matrix, NetworkTypeLabels, sub_to_main_type, isSubType)
	
	
	# for i in range(10):
	# 	cm, NetworkTypeLabels = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType, "SMOTE")
		
if __name__ == '__main__':
	main()