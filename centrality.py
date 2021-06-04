#%reset to clear all variables
# import sys
# sys.modules[__name__].__dict__.clear()
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
%matplotlib inline
import numpy as np
from collections import Counter
from pyproj import Proj, transform
import itertools
from itertools import groupby
from operator import itemgetter
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import sklearn.cluster as cluster
import seaborn as sns
import time
from sklearn.neighbors import NearestNeighbors
from geopy.distance import great_circle
from shapely.geometry import Point, MultiPoint
from matplotlib import cm
from scipy.spatial import ConvexHull
from functools import reduce
import networkx as nx
import geopandas as gpd
import itertools
import copy
import osmnx as ox


sample = pd.read_csv('D:/BILAL/flood_project/daejeon_all_links.csv')


#converting our data into graph data
xx = sample.drop_duplicates(subset=['LINK_ID'])
# xx = xx.sample(n = 1000)
edgelist = xx[['START_NODE_ID','END_NODE_ID','Trvl Tm']]
nodelist = xx[['LINK_ID','O_x','O_y']]
g = nx.Graph()
# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())
nx.info(g)
g.nodes()
len(g.nodes())

nx.draw(g, node_size=10)

# nx.draw_networkx
#Analysis
#Degree of Centrality
#1.degree of connection
nx.degree(g)
nx.degree(g, 2920012103)
#2. most influential
most_influential = nx.degree_centrality(g)
sorted(nx.degree_centrality(g))
for w in sorted(most_influential, key = most_influential.get, reverse=True):
    print(w,most_influential[w])
#3. Most important connection
nx.eigenvector_centrality(g, max_iter=600)
most_imp_link = nx.eigenvector_centrality(g, max_iter=600)
for w in sorted(most_imp_link, key = most_imp_link.get, reverse=True):
    print(w,most_imp_link[w])
#shortest path
nx.shortest_path(g,1870014700, 1870013600, weight = 'Trvl Tm')
#4. between centrality
best_connector = nx.betweenness_centrality(g)
for w in sorted(best_connector, key = best_connector.get, reverse=True):
    print(w,best_connector[w])


df = pd.DataFrame(dict(
    #number of nodes connected to the node
    DEGREE = dict(g.degree),
    #nodes connected vs totlal possible connections
    DEGREE_CENTRALITY = nx.degree_centrality(g),
    #most important connection
    EIGENVECTOR = nx.eigenvector_centrality(g,max_iter=600, weight = 'Trvl Tm'),
    PAGERANK = nx.pagerank(g, weight = 'LINK_LEN'),
    #how close the other nodes are
    # Closeness centrality [1] of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes.
    CLOSENESS_CENTRALITY = nx.closeness_centrality(g, distance = 'Trvl Tm'),
    #how many times same node is traversed on shortest path
    BETWEENNESS_CENTRALITY = nx.betweenness_centrality(g, weight = 'Trvl Tm'),
    # BETWEENNESS_EDGE = nx.edge_betweenness_centrality(g, weight = 'LINK_LEN')
))

df['START_NODE_ID'] = df.index
df2 = pd.merge(sample,df, how='left', on= 'START_NODE_ID')
df3 = df2.drop_duplicates(subset=['LINK_ID'])
# df3.to_csv('D:/BILAL/centrality/centrality.py.csv', index = False)




