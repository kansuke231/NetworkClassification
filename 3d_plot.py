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


def plot_3d(data):

    ts = [t for f1,f2,f3,t in data]

    colors = ["y","r","c","b","g","k","m"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t,c in zip(set(ts),colors):
        xs = [f1 for f1,f2,f3,network_type in data if network_type == t]
        ys= [f2 for f1,f2,f3,network_type in data if network_type == t]
        zs = [f3 for f1,f2,f3,network_type in data if network_type == t] 
        ax.plot(xs, ys, zs,"o",c=c,label=t)
    
    ax.set_xlabel('Modularity')
    ax.set_ylabel('Mean Geodesic Distance')
    ax.set_zlabel('Clustering Coefficient')
    ax.legend(loc = 'upper left')
    plt.draw()
    plt.show()


def main():
	params = sys.argv
	filepath = params[1]
	network_dict = data_read(filepath, "NetworkType","ClusteringCoefficient","MeanGeodesicPath","Modularity")
	network_tuple = [tuple(map(lambda x: float(x) if isFloat(x) else x,network_dict[k].values())) for k in network_dict]
	plot_3d(network_tuple)

if __name__ == '__main__':
	main()