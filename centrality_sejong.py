# %%
from itertools import groupby
from functools import reduce
import itertools
import osmnx as ox
import pandas as pd
import numpy as np
import networkx as nx

#%%
#function to convert the degree to straight distance

def haversine_vectorize(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    newlon = lon2 - lon1
    newlat = lat2 - lat1

    haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

    dist = 2 * np.arcsin(np.sqrt(haver_formula))
    km = 6367 * dist #6367 for distance in KM for miles use 3958
    return km

#%%
sample1 = pd.read_excel('D:/BILAL/centrality/BUSROUTE.xlsx', sheet_name='테이블')
sample2 = pd.read_excel('D:/BILAL/centrality/BUSROUTE_STOP_DETAIL(노선별 정류장 시퀀스).xlsx', sheet_name='테이블')
sample2 = pd.merge(sample2,sample1[['ROUTESECT_GROUP_ID','ROUTE_ID']],how='inner',on=['ROUTESECT_GROUP_ID'])
sample3 = pd.read_excel('D:/BILAL/centrality/BISNODE_NEW(버스정류장).xlsx', sheet_name='테이블')

sample3['NODE_ID'] = sample3['NODE_ID'].astype(str)
sample2['NODE_ID'] = sample2['NODE_ID'].astype(str)
sample2['end_node'] = sample2['NODE_ID'].shift(-1)
sample2['len'] = sample2['SECT_DISTANCE'].shift(-1)

df = sample2.groupby("ROUTESECT_GROUP_ID", as_index=False).apply(lambda x: x.iloc[:-1])

df2 = pd.merge(df, sample3[['NODE_ID','LAT','LNG']], how='left', left_on='NODE_ID', right_on='NODE_ID')
df3 = pd.merge(df2, sample3[['NODE_ID','LAT','LNG']], how='left', left_on='end_node', right_on='NODE_ID')
#%%
df3['straight_d'] = haversine_vectorize(df3['LNG_x'],df3['LAT_x'],df3['LNG_y'],df3['LAT_y']) 


#%%
#converting our data into graph data
xx = df3
# xx = xx.sample(n = 1000)
edgelist = xx[['NODE_ID_x','end_node','len']]
nodelist = xx[['NODE_ID_x','LNG_x','LAT_x']]
#%%
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
# nx.degree(g)
# nx.degree(g, 293001084)
# #2. most influential
# most_influential = nx.degree_centrality(g)
# sorted(nx.degree_centrality(g))
# for w in sorted(most_influential, key = most_influential.get, reverse=True):
#     print(w,most_influential[w])
# #3. Most important connection
# nx.eigenvector_centrality(g, max_iter=600)
# most_imp_link = nx.eigenvector_centrality(g, max_iter=600)
# for w in sorted(most_imp_link, key = most_imp_link.get, reverse=True):
#     print(w,most_imp_link[w])
# #4. between centrality
# best_connector = nx.betweenness_centrality(g)
# for w in sorted(best_connector, key = best_connector.get, reverse=True):
#     print(w,best_connector[w])

#%%
df = pd.DataFrame(dict(
    #number of nodes connected to the node
    DEGREE = dict(g.degree),
    #nodes connected vs totlal possible connections
    DEGREE_CENTRALITY = nx.degree_centrality(g),
    #most important connection
    EIGENVECTOR = nx.eigenvector_centrality(g,max_iter=600, weight = 'len'),
    PAGERANK = nx.pagerank(g, weight = 'len'),
    #how close the other nodes are
    # Closeness centrality [1] of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes.
    CLOSENESS_CENTRALITY = nx.closeness_centrality(g, distance = 'len'),
    #how many times same node is traversed on shortest path
    BETWEENNESS_CENTRALITY = nx.betweenness_centrality(g, weight = 'len'),
    # BETWEENNESS_EDGE = nx.edge_betweenness_centrality(g, weight = 'LINK_LEN')
    # SHORTEST_DIST = nx.shortest_path_length(g, weight='len')
))

df['NODE_ID'] = df.index
# dff = pd.merge(df3,df, how='left', on= 'NODE_ID')
dff = pd.merge(df3, df, how='left', left_on='NODE_ID_x', right_on='NODE_ID')


#%%
ride = pd.read_excel('D:/BILAL/centrality/volumedata/BUS_CARD_BRD_DEAL.xlsx')
take = pd.read_excel('D:/BILAL/centrality/volumedata/BUS_CARD_ALIT_DEAL.xlsx')

sum_ride = ride.groupby(['BRD_STTN_ID']).agg({'USER_CNT': 'sum'})
sum_ride.reset_index(inplace = True)
sum_ride.columns = ['NODE_ID', 'USER_CNT']

take_ride = take.groupby(['ALIT_STTN_ID']).agg({'USER_CNT': 'sum'})
take_ride.reset_index(inplace = True)
take_ride.columns = ['NODE_ID', 'USER_CNT_t']

sum_ride['NODE_ID'] = sum_ride['NODE_ID'].astype(str)
take_ride['NODE_ID'] = take_ride['NODE_ID'].astype(str)

total_count = pd.merge(sum_ride,take_ride, how='outer', on='NODE_ID')
total_count.fillna(0, inplace = True)
total_count['volume'] = total_count['USER_CNT'] + total_count['USER_CNT_t']

dfff = pd.merge(dff,total_count,how='left', left_on='NODE_ID', right_on='NODE_ID')
# dfff.to_csv('D:/BILAL/centrality/centrality_sejong.py.csv', index = False)

#%%
# trip chain data

trip_trail = pd.read_csv('D:/BILAL/centrality/trip chain data/totalAgentBusTrailEvent.csv')
bus_agent = pd.read_csv('D:/BILAL/centrality/trip chain data/totalBusAgent.csv')

# %%
ex_sum = ride.groupby(['BRD_STTN_ID']).agg({'TRSF_CNT': 'sum'})
ex_sum.reset_index(inplace = True)
ex_sum.columns = ['NODE_ID', 'TRSF_CNT']
ex_sum['NODE_ID'] = ex_sum['NODE_ID'].astype(str)

ex_data = pd.merge(dff,ex_sum,how='left', left_on='NODE_ID', right_on='NODE_ID')
singcha_data = pd.merge(dff,sum_ride,how='left', left_on='NODE_ID', right_on='NODE_ID')
hacha_data = pd.merge(dff,take_ride,how='left', left_on='NODE_ID', right_on='NODE_ID')

ex_data.to_csv('D:/BILAL/centrality/ex_data.csv', index = False)
singcha_data.to_csv('D:/BILAL/centrality/singcha_data.csv', index = False)
hacha_data.to_csv('D:/BILAL/centrality/hacha_data.csv', index = False)


# %%
import pandas as pd
schedule = pd.read_excel('D:/BILAL/centrality/bus time table/BUS_TIME_TABLE_정렬_시간수정.xlsx', sheet_name='테이블')

# %%
schedule.loc[schedule['WEEK_OPERATION_FLAG']==2,'WEEK_OPERATION_FLAG'] = 1

# %%
weekday_sum = schedule.groupby(['ROUTE_ID']).agg({'WEEK_OPERATION_FLAG': 'sum'})
sat_sum = schedule.groupby(['ROUTE_ID']).agg({'SAT_OPERATION_FLAG': 'sum'})
holiday_sum = schedule.groupby(['ROUTE_ID']).agg({'HOLIDY_OPERATION_FLAG': 'sum'})
weekday_sum.reset_index(inplace=True)
sat_sum.reset_index(inplace=True)
holiday_sum.reset_index(inplace=True)

# %%
dfs = [weekday_sum, sat_sum, holiday_sum]
schedue_sum = reduce(lambda left,right: pd.merge(left,right,on='ROUTE_ID'), dfs)

# %%
col_list= list(schedue_sum)
col_list.remove('ROUTE_ID')
schedue_sum['total_freq'] = schedue_sum[col_list].sum(axis=1)

# %%
schedue_com = pd.merge(schedue_sum[['ROUTE_ID','WEEK_OPERATION_FLAG','SAT_OPERATION_FLAG','HOLIDY_OPERATION_FLAG','total_freq']],dfff[['ROUTE_ID','NODE_ID_x','end_node','LAT_x', 'LNG_x','LAT_y', 'LNG_y']],how='left', right_on='ROUTE_ID', left_on='ROUTE_ID')

# %%
xx = schedue_com.groupby(['NODE_ID_x']).agg({'total_freq': 'sum', 'WEEK_OPERATION_FLAG': 'sum','SAT_OPERATION_FLAG': 'sum','HOLIDY_OPERATION_FLAG': 'sum'})
xx.reset_index(inplace=True)
# %%
df_mod = dfff.drop_duplicates(subset='NODE_ID_x')
schedue_com_new = pd.merge(xx[['NODE_ID_x','total_freq','WEEK_OPERATION_FLAG','SAT_OPERATION_FLAG','HOLIDY_OPERATION_FLAG']],df_mod[['ROUTE_ID','NODE_ID_x','end_node','LAT_x', 'LNG_x','LAT_y', 'LNG_y']],how='inner', left_on='NODE_ID_x', right_on='NODE_ID_x')

# %%
schedue_com_new.to_csv('D:/BILAL/centrality/schedue_com_new.py.csv', index = False)

#%%
#since importance is given to shorter distance, we will take the inverse of the total_frequency of buses to account for it
schedue_com_new['total_freq'] = 1/schedue_com_new['total_freq']

# %%
#centrality based on bus frequencies
edgelist = schedue_com_new[['NODE_ID_x','end_node','total_freq']]
nodelist = schedue_com_new[['NODE_ID_x','LNG_x','LAT_x']]
g = nx.Graph()
# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())
nx.info(g)
g.nodes()

#%%
df_seq = pd.DataFrame(dict(
    #number of nodes connected to the node
    DEGREE = dict(g.degree),
    #nodes connected vs totlal possible connections
    DEGREE_CENTRALITY = nx.degree_centrality(g),
    #most important connection
    EIGENVECTOR = nx.eigenvector_centrality(g,max_iter=1000, weight = 'total_freq'),
    PAGERANK = nx.pagerank(g, weight = 'total_freq'),
    #how close the other nodes are
    # Closeness centrality [1] of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes.
    CLOSENESS_CENTRALITY = nx.closeness_centrality(g, distance = 'total_freq'),
    #how many times same node is traversed on shortest path
    BETWEENNESS_CENTRALITY = nx.betweenness_centrality(g, weight = 'total_freq'),
    # BETWEENNESS_EDGE = nx.edge_betweenness_centrality(g, weight = 'LINK_LEN')
    # SHORTEST_DIST = nx.shortest_path_length(g, weight='len')
))

df_seq['NODE_ID_x'] = df_seq.index
# dff = pd.merge(df3,df, how='left', on= 'NODE_ID')
dff_seq = pd.merge(schedue_com_new, df_seq, how='left', left_on='NODE_ID_x', right_on='NODE_ID_x')
# %%
dff_seq.to_csv('D:/BILAL/centrality/busFrequency_centrality.py.csv', index = False)

#%%
tripchain = pd.read_excel('D:/BILAL/centrality/trip chain data/BUS CARD DATA_20210527.xlsx')

# %%
#removing no hacha travelers or trips
# tripchain = tripchain[tripchain['CARD_ALIT_STTN_ID'].notnull(),]
tripchain = tripchain[tripchain['CARD_ALIT_STTN_ID'].notna()]

#%%
tripchain['CARD_BRD_STTN_ID'] = tripchain['CARD_BRD_STTN_ID'].astype(str)
tripchain['CARD_ALIT_STTN_ID'] = tripchain['CARD_ALIT_STTN_ID'].astype(str)
tripchain['ODpair'] = tripchain['CARD_BRD_STTN_ID'] + '-' + tripchain['CARD_ALIT_STTN_ID']

# %%
ODVolume = tripchain.groupby(['ODpair']).agg({'USER_CNT': 'sum'})
ODVolume.reset_index(inplace=True)

#%%
ODVolume[['NODE_ID_x','end_node']] = ODVolume['ODpair'].str.split('-', 1, expand=True)

#%%
ODVolume[['end_node','miss']] = ODVolume['end_node'].str.split('.', 1, expand=True)
ODVolume = ODVolume.drop("miss", axis=1)
ODVolume = ODVolume.drop("NODE_ID", axis=1)

# %%
ODVolume2 = pd.merge(ODVolume, sample3[['NODE_ID','LAT','LNG']], how='left', left_on='NODE_ID_x', right_on='NODE_ID')
ODVolume2 = ODVolume2.drop("NODE_ID", axis=1)
ODVolume3 = pd.merge(ODVolume2, sample3[['NODE_ID','LAT','LNG']], how='left', left_on='end_node', right_on='NODE_ID')
ODVolume3 = ODVolume3.drop("NODE_ID", axis=1)

# %%
#centrality based on bus frequencies
edgelist = ODVolume3[['NODE_ID_x','end_node','USER_CNT']]
nodelist = ODVolume3[['NODE_ID_x','LNG_x','LAT_x']]
g = nx.Graph()
# Add edges and edge attributes
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1], attr_dict=elrow[2:].to_dict())
nx.info(g)
g.nodes()

#%%
df_seq = pd.DataFrame(dict(
    #number of nodes connected to the node
    DEGREE = dict(g.degree),
    #nodes connected vs totlal possible connections
    DEGREE_CENTRALITY = nx.degree_centrality(g),
    #most important connection
    EIGENVECTOR = nx.eigenvector_centrality(g,max_iter=2000, weight = 'USER_CNT'),
    PAGERANK = nx.pagerank(g, weight = 'USER_CNT'),
    #how close the other nodes are
    # Closeness centrality [1] of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes.
    CLOSENESS_CENTRALITY = nx.closeness_centrality(g, distance = 'USER_CNT'),
    #how many times same node is traversed on shortest path
    BETWEENNESS_CENTRALITY = nx.betweenness_centrality(g, weight = 'USER_CNT'),
    # BETWEENNESS_EDGE = nx.edge_betweenness_centrality(g, weight = 'LINK_LEN')
    # SHORTEST_DIST = nx.shortest_path_length(g, weight='len')
))

df_seq['NODE_ID_x'] = df_seq.index
# dff = pd.merge(df3,df, how='left', on= 'NODE_ID')
dff_seq = pd.merge(ODVolume3, df_seq, how='left', left_on='NODE_ID_x', right_on='NODE_ID_x')
# %%
dff_seq.to_csv('D:/BILAL/centrality/tripcahin_Centrality.py.csv', index = False)
# %%
