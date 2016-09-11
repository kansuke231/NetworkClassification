from misc import init
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from plot import plot_feature_importance
from sklearn.metrics import roc_auc_score

def convert_one_to_many(X,Y,label):

	X_converted = []
	Y_converted = []

	for x,y in zip(X,Y):
		X_converted.append(x)
		if y == label:
			Y_converted.append(1)
		else:
			Y_converted.append(0)

	return X_converted, Y_converted


def split_train_test(X,Y):
	X = np.array(X)
	Y = np.array(Y)
	sss = StratifiedShuffleSplit(Y, 3, test_size=0.3, random_state=0)
	for train_index, test_index in sss:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]

	return X_train, X_test, y_train, y_test



def one_to_many_classification(X_train, X_test, y_train, y_test, feature_order):
	random_forest = RandomForestClassifier()
	random_forest.fit(X_train,y_train)
	accuracy = random_forest.score(X_test,y_test)
	y_pred = random_forest.predict(X_test)
	feature_importances = sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), feature_order), reverse=True)
	AUC = roc_auc_score(y_test, y_pred)
	print "Accuracy: %f"%accuracy
	print "AUC: %f"%AUC
	print "Feature Importance: ", feature_importances
	return accuracy, feature_importances, roc_auc_score(y_test, y_pred)

def many_classifications(X, Y, sub_to_main_type, feature_order, N):
	
	list_important_features = []
	list_accuracies = []
	list_auc = []
	for i in range(N):
		X_train, X_test, y_train, y_test = split_train_test(X, Y)
		accuracy, feature_importances, auc = one_to_many_classification(X_train, X_test, y_train, y_test, feature_order)
		list_important_features.append(feature_importances)
		list_accuracies.append(accuracy)
		list_auc.append(auc)

	return list_accuracies, list_important_features, list_auc

def main():
	#column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity","DegreeAssortativity","MeanGeodesicDistance","Diameter","MGD/Diameter",
	#			    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]

	column_names = ["NetworkType","SubType","ClusteringCoefficient","DegreeAssortativity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	
	isSubType = True
	at_least = 6
	X,Y, sub_to_main_type, feature_order = init("features.csv", column_names, isSubType, at_least)
	N = 100
	X_converted, Y_converted = convert_one_to_many(X,Y,"Subway")
	list_accuracies, list_important_features, list_auc = many_classifications(X_converted, Y_converted, sub_to_main_type, feature_order, N)
	print "average accuracy: %f"%(float(sum(list_accuracies))/float(N))
	print "average AUC: %f"%(float(sum(list_auc))/float(N))
	plot_feature_importance(list_important_features, feature_order)


	

if __name__ == '__main__':
	main()