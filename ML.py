import sys
import csv
import math
from collections import Counter
from itertools import permutations

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.utils import check_X_y

from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.over_sampling import RandomOverSampler

def data_read(filepath, *features):
	"""
	Read a csv file (features.csv).
	"""
	network_dict = {}
	with open(filepath, 'rb') as f:
		reader = csv.DictReader(f)
		for row in reader:

			filtered = dict((k,v) for k,v in row.items() if all([(k in features), v, (v != "nan")]))
			
			# if filtered lacks some feautres, e.g. not calculated yet.
			if len(filtered) != len(features):
				continue
			elif row["NetworkType"] == "Synthetic":
				continue
			#elif row["NetworkType"] == "Biological":
			else:
				gml_name = row[".gmlFile"]
				network_dict[gml_name] = filtered

	return network_dict


def filter_float(network_dict):
	"""
	Make a string of float in the dictionary into float.
	"""
	for gml_name in network_dict:
		for e in network_dict[gml_name]:
			if not((e == "NetworkType") or (e == "SubType")):
				network_dict[gml_name][e] = float(network_dict[gml_name][e])
	return network_dict

def XY_generator(network_dict):
	"""
	Separate labels and features values for scikit-learn algorithms.
	"""
	
	X = []
	Y = []

	for gml_name in network_dict:
		d = network_dict[gml_name] # d for dictionary.
		Y.append((gml_name, d["NetworkType"], d["SubType"]))
		X.append(dict((k,v) for k,v in d.items() if not(k == "NetworkType" or k == "SubType")))

	return X,Y

def XY_filter_unpopular(X, Y, threshold):
	"""
	filters out the elements which are unpopular (i.e. # of ys is below threshold)
	"""
	counts = Counter(Y)
	popular = [elem for (elem, count) in filter(lambda (e,c): c > threshold ,counts.most_common())]
	return np.concatenate(tuple(X[Y == p] for p in popular),axis=0),\
		   np.concatenate(tuple(Y[Y == p] for p in popular),axis=0)
	


#------------------------------ MultiClass Classification ------------------------------
def multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType):
	"""
	This functions returns a confusion matrix and NetworkTypeLabels
	"""

	if isSubType:
		NetworkTypeLabels = sorted(list(set(Y)), key=lambda sub_type: sub_to_main_type[sub_type])
	else:
		NetworkTypeLabels = sorted(list(set(Y))) # for NetworkType
	
	sss = StratifiedShuffleSplit(Y, 3, test_size=0.35, random_state=0)
	for train_index, test_index in sss:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]

	random_over = RandomOverSampler()
	random_x, random_y = random_over.fit_transform(X_train, y_train)


	# TODO: Implement an over-sampling using SMOTE. 
	# First, determine which class is the largest amongst others.
	# Then over-sample the other classes according to the ratio to the largest one.
	# Then combine the X and Y. Don't forget to eliminate duplications! (maybe use set??)

	#majority = 


	random_forest = RandomForestClassifier()
	random_forest.fit(random_x,random_y)

	print "Feature Importance"
	print sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), feature_names), reverse=True)
	print "----------------------------------------------------"
	print "prediction: %f"%random_forest.score(X_test,y_test)

	y_pred = random_forest.predict(X_test)
	cm = confusion_matrix(y_test, y_pred, labels=NetworkTypeLabels)
	return cm, NetworkTypeLabels

def plot_confusion_matrix(cm, NetworkTypeLabels, sub_to_main_type, isSubType):
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_filtered = map(lambda ax: map(lambda val: 0.0 if math.isnan(val) else val, ax),cm_normalized)

    f, ax = plt.subplots()
    im = ax.imshow(cm_normalized_filtered, interpolation='nearest', cmap=plt.cm.Blues)

    if isSubType:
    	prev = sub_to_main_type[NetworkTypeLabels[0]]
    	i = -0.5
    	for t in NetworkTypeLabels:
    		if prev != sub_to_main_type[t]:
    			ax.axhline(i)
    			ax.axvline(i)
    			prev = sub_to_main_type[t]
    		i += 1

    dim = len(cm)
    for i in range(dim):
    	for j in range(dim):
    		if cm[i][j] != 0.0:
    			ax.text(j, i, cm[i][j], va='center', ha='center', color = "r")
    
    #ax.set_title("Confusion Matrix")
    f.colorbar(im)
    tick_marks = np.arange(len(NetworkTypeLabels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(NetworkTypeLabels, rotation=90)
    ax.set_yticklabels(NetworkTypeLabels)
    f.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.show()

#------------------------------ Binary Classification ------------------------------

def AUC(features, Y):
	"""
	Returns an AUC score for a given pair of classes.
	"""
	v = DictVectorizer(sparse=False)
	X = v.fit_transform(features)
	Y = np.array(Y)
	#print "positive: %d negative: %d"%(np.count_nonzero(Y),len(Y)-np.count_nonzero(Y))
	ratio = float(np.count_nonzero(Y==0))/float(np.count_nonzero(Y==1))	
	
	# if ratio < 1.0:
	# 	ratio = 0

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
		if len([1 for l in ls if l == 1]) < len([1 for l in ls if l == 0]):
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


def plot_distance_matrix(distance_m, NetworkTypeLabels, sub_to_main_type, isSubType):
    
    f, ax = plt.subplots()
    im = ax.imshow(distance_m, interpolation='nearest', cmap=plt.cm.Blues)

    if isSubType:
    	prev = sub_to_main_type[NetworkTypeLabels[0]]
    	i = -0.5
    	for t in NetworkTypeLabels:
    		if prev != sub_to_main_type[t]:
    			ax.axhline(i)
    			ax.axvline(i)
    			prev = sub_to_main_type[t]
    		i += 1

    # dim = len(distance_m)
    # for i in range(dim):
    # 	for j in range(dim):
    # 		#if distance_m[i][j] != 0.0:
    # 		ax.text(j, i, round(distance_m[i][j],2), va='center', ha='center', color = "r")
    
    f.colorbar(im)
    tick_marks = np.arange(len(NetworkTypeLabels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(NetworkTypeLabels, rotation=90,fontsize=10)
    ax.set_yticklabels(NetworkTypeLabels, fontsize=10)
    #f.tight_layout()
    plt.show()

#------------------------------ LDA ------------------------------
def plot_scikit_lda(X, Y):
    ts = set(Y)
    values = range(len(ts))
    jet = plt.get_cmap('jet') 
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(ts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    ax = plt.subplot(111)
    for idx,label in enumerate(ts):
    	colorVal = scalarMap.to_rgba(values[idx])

        plt.scatter(x=X[:,0][Y == label],
                    y=X[:,1][Y == label] * (-1), # flip the figure
                    color=colorVal,
                    alpha=0.8,
                    label=label
                    )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.1, 1.05),prop={'size':10})
    
    plt.tight_layout
    plt.show()

def plot_scikit_lda_3d(X, Y):
    ts = set(Y)
    values = range(len(ts))
    jet = plt.get_cmap('jet') 
    cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(ts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx,t in enumerate(ts):
    	colorVal = scalarMap.to_rgba(values[idx])
        ax.plot(X[:,0][Y == t], 
        		X[:,1][Y == t], 
        		X[:,2][Y == t],"o",c=colorVal,label=t,alpha=0.85)
    
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    #ax.set_zscale("log")
    ax.legend(loc = 'upper right',bbox_to_anchor=(1.1, 1.05),prop={'size':10})
    plt.draw()
    plt.show()

def LinearDiscriminantAnalysis(X,Y):
	sklearn_lda = LDA(n_components=3)
	X_lda_sklearn = sklearn_lda.fit_transform(X, Y)
	return X_lda_sklearn


def MDS_plot(distance_matrix, NetworkTypeLabels):
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(distance_matrix).embedding_
	clf = PCA(n_components=2)
	pos = clf.fit_transform(pos)

	print pos
	xs = [x for x,y in pos]
	ys = [y for x,y in pos]
	plt.scatter(xs, ys)
	for i,type_label in enumerate(NetworkTypeLabels):
		plt.annotate(type_label,xy = (xs[i], ys[i]))
	plt.axhline(0)
	plt.axvline(0)
	plt.show()
	
def matrix_clustering(D, leave_name):
	import scipy
	import pylab
	import scipy.cluster.hierarchy as sch
	
	# Compute and plot first dendrogram.
	fig = pylab.figure(figsize=(20,20))
	ax1 = fig.add_axes([0.00,0.1,0.2,0.6])
	Y = sch.linkage(D, method='centroid')
	Z1 = sch.dendrogram(Y, orientation='right')
	ax1.set_xticks([])
	ax1.set_yticks([])
	
	# Compute and plot second dendrogram.
	ax2 = fig.add_axes([0.3,0.71,0.6,0.05])
	Y = sch.linkage(D, method='centroid')
	Z2 = sch.dendrogram(Y)
	ax2.set_xticks([])
	ax2.set_yticks([])
	
	# Plot distance matrix.
	axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
	idx1 = Z1['leaves']
	idx2 = Z2['leaves']
	D = D[idx1,:]
	D = D[:,idx2]
	im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)

	# mapping from an index to an axis label (gml file name, NetworkType, SubType)
	# axis_labels = [leave_name[i] for i in idx1]

	# tick_marks = np.arange(len(axis_labels))
	# axmatrix.yaxis.set_label_position('right')
	# axmatrix.set_yticks(tick_marks)
	# axmatrix.set_yticklabels(axis_labels)
	# pylab.yticks(fontsize=7)
	
	# Plot colorbar.
	#axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
	#pylab.colorbar(im, cax=axcolor)
	fig.show()
	fig.savefig('dendrogram.png',bbox_inches='tight')

def main(analysis):
	network_dict = data_read("features.csv","NetworkType","SubType","ClusteringCoefficient","Modularity","MeanGeodesicDistance",\
							 "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6")

	feature_names = ["ClusteringCoefficient","MeanGeodesicPath","Modularity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	network_dict = filter_float(network_dict)
	features,labels = XY_generator(network_dict)

	v = DictVectorizer(sparse=False)
	X = v.fit_transform(features)

	sub_to_main_type = dict((SubType,NetworkType) for gml, NetworkType, SubType in labels)

	isSubType = True

	if isSubType:
		Y = np.array([SubType for gml, NetworkType, SubType in labels])
	else:
		Y = np.array([NetworkType for gml, NetworkType, SubType in labels])

	at_least = 4
	X,Y = XY_filter_unpopular(X, Y, at_least)
	
	# Branches of different analyses

	if analysis == "MultiClass":
		cm, NetworkTypeLabels = multiclass_classification(X, Y, sub_to_main_type, feature_names, isSubType)
		plot_confusion_matrix(cm, NetworkTypeLabels, sub_to_main_type, isSubType)

	elif analysis == "BinaryClass":
		distance_matrix, NetworkTypeLabels = binary_classification(features, labels, feature_names, isSubType=isSubType)
		plot_distance_matrix(distance_matrix, NetworkTypeLabels, sub_to_main_type, isSubType=isSubType)
		MDS_plot(distance_matrix, NetworkTypeLabels)

	elif analysis == "LDA":
		X_lda_sklearn = LinearDiscriminantAnalysis(X, Y)
		plot_scikit_lda_3d(X_lda_sklearn, Y)
		plot_scikit_lda(X_lda_sklearn, Y)
	
	elif analysis == "MatrixClustering":
		distance_matrix = euclidean_distances(X)
		matrix_clustering(distance_matrix, Y)
	



if __name__ == "__main__":
	param = sys.argv
	analysis_type = param[1]
	main(analysis_type)


