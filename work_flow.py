#!/Users/ikeharakansuke/env/bin/python
from __future__ import division
import math
from misc import init
import numpy as np
from multiclass import multiclass_classification
from plot import plot_confusion_matrix
from plot import plot_distance_matrix
from plot import matrix_clustering
from plot plot_feature_importance
from LDA import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cmx

import scipy.spatial.distance as ssd

def index_to_color(iterator):
	jet = plt.get_cmap('jet')
	cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(iterator))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	return lambda i: scalarMap.to_rgba(i)

def plot_feature_importance(Ls, feature_order):
	Ls = map(lambda x: map(lambda y: y[1],x), Ls)
	Ls = zip(*Ls) # Ls = [("ClustersingCoefficient", "ClustersingCoefficient", ..),]
	
	freq = {f: [] for f in feature_order}

	for freq_fs in Ls:
		for f in feature_order:
			freq[f].append(freq_fs.count(f))

	
	color_map = index_to_color(freq)

	iterate = sorted(list(freq.keys()),key=lambda x: x,reverse=True)

	first = iterate[0]
	colorVal = color_map(0)
	p = plt.bar(range(len(feature_order)), freq[first],  0.35, color=colorVal)
	prev = freq[first] # previous stack

	ps = [p] # storing axis objects
	who_is_dominant = [map(lambda x: (first,x),freq[first])]

	for i,k in enumerate(iterate[1:]):
		colorVal = color_map(i+1)
		p = plt.bar(range(len(feature_order)), freq[k],  0.35, color=colorVal, bottom=prev)
		who_is_dominant.append(map(lambda x: (k,x),freq[k]))
		prev = map(lambda x: x[0]+x[1] , zip(prev,freq[k]))
		ps.append(p)

	for rank in zip(*who_is_dominant):
		print sorted(rank, key=lambda x:x[1],reverse=True)

	plt.legend(ps,iterate, bbox_to_anchor=(1.12, 0.4),prop={'size':12})
	plt.xlabel('Feature Importance')
	plt.ylabel('Frequency')

	plt.show()

def sum_confusion_matrix(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod, N):
	accum_matrix, NetworkTypeLabels, accum_acc, feature_importances = multiclass_classification(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod)
	list_important_features = [feature_importances]
	for i in range(N - 1):
		cm, _, accuracy, feature_importances = multiclass_classification(X, Y, sub_to_main_type, feature_order, isSubType, samplingMethod)
		accum_matrix += cm
		accum_acc += accuracy
		list_important_features.append(feature_importances)
	return accum_matrix, NetworkTypeLabels, accum_acc, list_important_features


def make_symmetric(cm):
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	cm_normalized_filtered = map(lambda ax: map(lambda val: 0.0 if math.isnan(val) else val, ax),cm_normalized)
	N = len(cm_normalized_filtered)

	# make cm symmetric
	for i in range(N):
		for j in range(N):
			if i == j:
				cm_normalized_filtered[i][j] = 1
			else:
				maximum = max([cm_normalized_filtered[i][j], cm_normalized_filtered[j][i]])
				cm_normalized_filtered[i][j] = maximum
				cm_normalized_filtered[j][i] = maximum


	# make values into distance 
	for i in range(N):
		for j in range(N):
			cm_normalized_filtered[i][j] = 1 - cm_normalized_filtered[i][j]

	return np.asarray(cm_normalized_filtered)


def build_dendrogram(D, leave_name, sub_to_main_type):
    import pylab
    import scipy.cluster.hierarchy as sch
    Domains = list(set(sub_to_main_type.values()))
    color_map = index_to_color(Domains)
    fig = pylab.figure(figsize=(12, 10))
    Y = sch.linkage(D, method='complete')  # , method='centroid')
    Z1 = sch.dendrogram(Y, orientation='right', labels=leave_name)
    ax = plt.gca()
    ylbls = ax.get_ymajorticklabels()
    for lbl in ylbls:
    	domain = sub_to_main_type[lbl.get_text()]
    	index = Domains.index(domain)
    	lbl.set_color(color_map(index))

    fig.show()
    fig.savefig('only_dendrogram.png', bbox_inches='tight')

def main():
	column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity","DegreeAssortativity","MeanGeodesicDistance","Diameter","MGD/Diameter",
				    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]

	column_names = ["NetworkType","SubType","ClusteringCoefficient","DegreeAssortativity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	#column_names = column_names[:-6]
	#column_names.pop(3)
	#column_names = column_names[:-6]
	#column_names = ["NetworkType","SubType","ClusteringCoefficient","MeanGeodesicDistance","Modularity"]
	
	isSubType = True
	at_least = 6
	X,Y,sub_to_main_type, feature_order = init("features.csv", column_names, isSubType, at_least)
	N = 100
	
	Matrix, NetworkTypeLabels, sum_accuracy, list_important_features = sum_confusion_matrix(X, Y, sub_to_main_type, feature_order, isSubType, "SMOTE", N)
	average_matrix = np.asarray(map(lambda row: map(lambda e: e/N ,row), Matrix))
	print "average accuracy: %f"%(float(sum_accuracy)/float(N))
	plot_feature_importance(list_important_features, feature_order)
	plot_confusion_matrix(average_matrix, NetworkTypeLabels, sub_to_main_type, isSubType)
	dist_matrix = make_symmetric(average_matrix)
	#dist_matrix = ssd.squareform(dist_matrix)
	
	#plot_distance_matrix(dist_matrix, NetworkTypeLabels, sub_to_main_type, isSubType)
	build_dendrogram(dist_matrix, NetworkTypeLabels, sub_to_main_type)
		
if __name__ == '__main__':
	main()