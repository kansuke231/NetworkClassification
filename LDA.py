from misc import *
import sys
from plot import plot_scikit_lda_3d, plot_scikit_lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_extraction import DictVectorizer
import numpy as np

def LinearDiscriminantAnalysis(X,Y):
	sklearn_lda = LDA(n_components=3)
	X_lda_sklearn = sklearn_lda.fit_transform(X, Y)
	return X_lda_sklearn

def main(analysis):
		
	column_names = ["NetworkType","SubType","ClusteringCoefficient","Modularity",#"MeanGeodesicDistance",\
				    "m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	feature_names = ["ClusteringCoefficient","MeanGeodesicPath","Modularity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	isSubType = True
	at_least = 6
	X,Y,sub_to_main_type = init("features.csv", column_names, feature_names, isSubType, at_least)

	X_lda_sklearn = LinearDiscriminantAnalysis(X, Y)
	plot_scikit_lda_3d(X_lda_sklearn, Y)
	plot_scikit_lda(X_lda_sklearn, Y)




if __name__ == "__main__":
	param = sys.argv
	analysis_type = param[1]
	main(analysis_type)

