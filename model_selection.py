from misc import init
from plot import index_to_color
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

Synthesized = ["Scale Free", "ER Network", "Small World"]

def make_layers(bp_tuple_L, base_Ys):

	predicted_Ys = list(set(map(lambda x: x[1], bp_tuple_L)))

	Ys = sorted(list(set(base_Ys)))
	accum_dic = {k:[0 for i in range(len(Ys))] for k in predicted_Ys}

	for base, predicted in bp_tuple_L:
		accum_dic[predicted][Ys.index(base)]+=1

	return Ys, accum_dic


def plot_accumulation(Ys, accum_dic):
	
	iterate = list(accum_dic.keys())

	color_map = index_to_color(iterate)

	first = iterate[0]
	colorVal = color_map(0)
	p = plt.barh(range(len(Ys)), accum_dic[first],  0.35, color=colorVal)
	
	prev = accum_dic[first] # previous stack
	ps = [p] # storing axis objects
	
	for i,k in enumerate(iterate[1:]):
		colorVal = color_map(i+1)
		p = plt.barh(range(len(Ys)), accum_dic[k],  0.35, color=colorVal, left=prev)
		
		prev = map(lambda x: x[0]+x[1] , zip(prev,accum_dic[k]))
		ps.append(p)
	
	plt.legend(ps,iterate, bbox_to_anchor=(1.12, 0.4),prop={'size':12})
	plt.yticks(range(len(Ys)), Ys)
	plt.ylabel('Base Classes')
	plt.xlabel('Frequency')
	plt.show()


def separator(X,Y):
	"""
	Separates Synthesized classes (network models) from real-world network classes
	"""
	real_X = []
	real_Y = []
	synthesized_X = []
	synthesized_Y = []

	for x,y in zip(X,Y):
		if y in Synthesized:
			synthesized_X.append(x)
			synthesized_Y.append(y)
		else:
			real_X.append(x)
			real_Y.append(y)

	return real_X, real_Y, synthesized_X, synthesized_Y

def real_to_model(real_X, real_Y, synthesized_X, synthesized_Y):
	"""
	Train on the real-world networks, classify synthesized networks
	"""
	random_forest = RandomForestClassifier()
	random_forest.fit(real_X, real_Y)
	y_pred = random_forest.predict(synthesized_X)
	return zip(y_pred, synthesized_Y)




def main():
	column_names = ["NetworkType","SubType","ClusteringCoefficient","DegreeAssortativity","m4_1","m4_2","m4_3","m4_4","m4_5","m4_6"]
	
	isSubType = True
	at_least = 1
	X,Y, sub_to_main_type, feature_order = init("features.csv", column_names, isSubType, at_least)
	N = 100
	#real_X, real_Y, synthesized_X, synthesized_Y = separator(X,Y)
	bp_tuple_L = real_to_model(*separator(X,Y))
	Ys, accum_dic =  make_layers(bp_tuple_L, Y)
	plot_accumulation(Ys, accum_dic)


if __name__ == '__main__':
	main()