import csv
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer

def data_read(filepath, *features, **kwargs):
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

			# below is for extracting only specific kinds of networks
			elif kwargs:
				if row["NetworkType"] in kwargs["types"]:
					gml_name = row[".gmlFile"]
					network_dict[gml_name] = filtered

			else:
				gml_name = row[".gmlFile"]
				network_dict[gml_name] = filtered

	return network_dict


def isFloat(x):
	try:
		float(x)
		return True
	except ValueError:
		return False


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


def filter_float(network_dict):
	"""
	Make a string of float in the dictionary into float.
	"""
	for gml_name in network_dict:
		for e in network_dict[gml_name]:
			if not((e == "NetworkType") or (e == "SubType")):
				network_dict[gml_name][e] = float(network_dict[gml_name][e])
	return network_dict


def XY_filter_unpopular(X, Y, threshold):
	"""
	filters out the elements which are unpopular (i.e. # of ys is below threshold)
	"""
	counts = Counter(Y)
	popular = [elem for (elem, count) in filter(lambda (e,c): c > threshold ,counts.most_common())]
	return np.concatenate(tuple(X[Y == p] for p in popular),axis=0),\
		   np.concatenate(tuple(Y[Y == p] for p in popular),axis=0)


def init(filepath, column_names, feature_names, isSubType, at_least):
	network_dict = data_read(filepath, *column_names)

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

	X,Y = XY_filter_unpopular(X, Y, at_least)

	return X,Y,sub_to_main_type
	



