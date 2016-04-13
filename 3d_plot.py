import sys
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def isFloat(x):
	try:
		float(x)
		return True
	except ValueError:
		return False

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


def sort_by_feature(network_dict, feature_names):
	result = []
	for k in network_dict.keys():
		each = []
		for name in feature_names:
			entry = network_dict[k][name]
			if isFloat(entry):
				each.append(float(entry))
			else:
				each.append(entry)
		result.append(tuple(each))
	return result

def plot_3d(data, feature_names):
    ts = [t for t,f1,f2,f3 in data]

    colors = ["y","r","c","b","g","k","m"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t,c in zip(set(ts),colors):
        xs = [f1 for network_type,f1,f2,f3 in data if network_type == t]
        ys = [f2 for network_type,f1,f2,f3 in data if network_type == t]
        zs = [f3 for network_type,f1,f2,f3 in data if network_type == t]
        
        ax.plot(xs, ys, zs,"o",c=c,label=t)
    
    ax.set_xlabel(feature_names[1])
    ax.set_ylabel(feature_names[2])
    ax.set_zlabel(feature_names[3])
    ax.legend(loc = 'upper left')
    plt.draw()
    plt.show()


def main():
	params = sys.argv
	filepath = params[1]
	feature_names = ["NetworkType","Modularity","ClusteringCoefficient","MeanGeodesicDistance"]
	#feature_names = ["NetworkType","m4_1","m4_3","m4_6"]
	network_dict = data_read(filepath, *feature_names)
	
	network_tuple = sort_by_feature(network_dict, feature_names)
	plot_3d(network_tuple, feature_names)

if __name__ == '__main__':
	main()