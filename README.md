# NetworkClassification

This repository contains the code for my M.S. thesis research titled
 "Structure of Complex Networks across Domains."

## Data
First thing first, we need data in order to analyze something. Code in this repo
 assumes the data, namely features of networks, is stored in a csv file named
 "features.csv" which looks like this:

```
.gmlFile,NetworkType,SubType,NumberOfNodes,NumberOfEdges,MeanDegree,MeanGeodesicDistance,...
some.gml,Social,Offline Social,62,152,4.9032,2.9455,...
another.gml,Social,Offline Social,62,152,4.9032,2.9455,...
```

## Analyses
The two main tasks in the research are: 1. finding which features make
a specific category of networks (i.e. online social networks) "stand out"
among others; 2. finding out which pairs of network categories are often misclassified
by a machine learning classifier based solely on the structural features and their implication in the process,
constraint and growing mechanism of those networks (if two distinct categories of networks are misclassified often, are their similar in some sense?).

The first task is done with the script named `one_by_many.py` and the second one
is done with `multi_run.py`

If one wants to see how the data points in a 2D or 3D feature space, `plot.py` contains
some code for 2D and 3D visualization.





