import csv

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