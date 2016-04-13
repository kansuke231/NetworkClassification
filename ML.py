import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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


def plot_confusion_matrix_new(cm_normalized, cm, NetworkTypeLabels, sub_to_main_type, isSubType, cmap=plt.cm.Blues):
    
    f, ax = plt.subplots()
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

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

def classification(features, labels, feature_names, isSubType):
	
	v = DictVectorizer(sparse=False)
	sub_to_main_type = dict((SubType,NetworkType) for gml, NetworkType, SubType in labels)

	if isSubType:
		Y = [SubType for gml, NetworkType, SubType in labels]
		NetworkTypeLabels = sorted(list(set(Y)), key=lambda sub_type: sub_to_main_type[sub_type])
	else:
		Y = [NetworkType for gml, NetworkType, SubType in labels]
		NetworkTypeLabels = sorted(list(set(Y))) # for NetworkType

	X = v.fit_transform(features)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

	random_forest = RandomForestClassifier()
	random_forest.fit(X_train,y_train)

	print "Feature Importance"
	print sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), feature_names), reverse=True)
	print "----------------------------------------------------"
	print "prediction: %f"%random_forest.score(X_test,y_test)

	y_pred = random_forest.predict(X_test)
	cm = confusion_matrix(y_test, y_pred, labels=NetworkTypeLabels)

	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	cm_normalized_filtered = map(lambda ax: map(lambda val: 0.0 if math.isnan(val) else val, ax),cm_normalized)
	plot_confusion_matrix_new(cm_normalized_filtered, cm, NetworkTypeLabels, sub_to_main_type, isSubType=isSubType)



def main():
	network_dict = data_read("features.csv","NetworkType","SubType","ClusteringCoefficient","MeanGeodesicPath",\
							 "Modularity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6")
	feature_names = ["ClusteringCoefficient","MeanGeodesicPath","Modularity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	network_dict = filter_float(network_dict)
	features,labels = XY_generator(network_dict)

	classification(features, labels, feature_names, isSubType=True)
	

if __name__ == "__main__":
	main()