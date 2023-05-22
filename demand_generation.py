import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit
from tqdm import tqdm
import pickle

child_zones_data = pd.read_excel('Greedy_OSM_test/Centroids_2931_v2.xlsx')
child_zones_data = child_zones_data.rename(columns={'Zone Id': 'zone_Id'})
od_parent_zones = pd.read_csv('Greedy_OSM_test/CARS (Peak Hour).csv', nrows=519, index_col=0)
od_parent_zones = od_parent_zones.iloc[:, :519]
np_array = np.array(od_parent_zones)
indices = np.nonzero(np_array)
od_child_zones = np.zeros((len(child_zones_data), len(child_zones_data)))
for i in range(len(indices[0])):
    orig_parent_zone = indices[0][i] + 1
    dest_parent_zone = indices[1][i] + 1
    orig_child_zones = np.array(child_zones_data[child_zones_data['Parent zone id'] == orig_parent_zone].zone_Id)
    dest_child_zones = np.array(child_zones_data[child_zones_data['Parent zone id'] == dest_parent_zone].zone_Id)
    orig_child_area = np.array(child_zones_data[child_zones_data['Parent zone id'] == orig_parent_zone].Z_Area_m2)
    dest_child_area = np.array(child_zones_data[child_zones_data['Parent zone id'] == dest_parent_zone].Z_Area_m2)
    proportion_matrix = orig_child_area[:, np.newaxis] + dest_child_area
    proportion_matrix = proportion_matrix / np.sum(proportion_matrix)
    od_child_zones[np.ix_(orig_child_zones - 1, dest_child_zones - 1)] \
        = np_array[orig_parent_zone - 1, dest_parent_zone - 1] * proportion_matrix

od_child_zones_df = pd.DataFrame(od_child_zones)
od_child_zones_df.index = list(range(1, len(child_zones_data) + 1))
od_child_zones_df.columns = list(range(1, len(child_zones_data) + 1))
od_child_zones_df.to_excel('Greedy_OSM_test/od_child_zones.xlsx')

graph = pickle.load(open('./Greedy_OSM_test/shp_to_pickle', 'rb'))
nodes = np.array(graph.nodes)
output_dict = {'Node_number': [],'Node Coordinates': [],'distance': []}
for idx,row in tqdm(child_zones_data.iterrows()):
    output_dict['Node_number'].append(row.zone_Id)
    distance_array = haversine_vector([(row.XX, row.YY)], nodes, Unit.METERS,comb=True)
    index_min = np.round(np.argmin(distance_array),2)
    output_dict['Node Coordinates'].append(tuple(nodes[index_min]))
    output_dict['distance'].append(distance_array[index_min][0])
output_df = pd.DataFrame(output_dict)
output_df[['Node_number','Node Coordinates']].to_csv('Greedy_OSM_test/child_zones_to_nodes.csv',index=False)