import datetime
import re
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
from haversine import haversine_vector, Unit
from itertools import tee
import builtins
import warnings
from collections import Counter
import random

warnings.filterwarnings("ignore")


# Function to make pairs of elements in a list
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return builtins.zip(a, b)


def read_GTFS_data(filename: str) -> None:
    # Reading GTFS data
    global routes_df, trips_df, stop_times_df, stops_df, fare_rules_df, fare_attributes
    with ZipFile(filename) as zip:
        with zip.open('London Underground GTFS/routes.txt') as routes:
            routes_df = pd.read_csv(routes)
        with zip.open('London Underground GTFS/trips.txt') as trips:
            trips_df = pd.read_csv(trips)
        with zip.open('London Underground GTFS/stop_times.txt') as stop_times:
            stop_times_df = pd.read_csv(stop_times)
        with zip.open('London Underground GTFS/stops.txt') as stops:
            stops_df = pd.read_csv(stops)
        with zip.open('London Underground GTFS/fare_rules.txt') as rules:
            fare_rules_df = pd.read_csv(rules)
        with zip.open('London Underground GTFS/fare_attributes.txt') as attributes:
            fare_attributes = pd.read_csv(attributes)
    return None


read_GTFS_data('./GTFS.zip')
fare_rules_df = fare_rules_df.drop_duplicates(subset=['fare_id', 'origin_id', 'destination_id', 'route_id'])
fare_rules_df = fare_rules_df.merge(fare_attributes, left_on='fare_id', right_on='fare_id', how='left')
avg_fare = fare_attributes['price'].mean()

# Reading 2019 od data
od_data_2019 = pd.read_csv('./Route_choice_Modelling/NBT19MTT3a_routechoice_tb_wf.csv')
lines_df = pd.read_excel('./Route_choice_Modelling/edited_NBT19_Definitions_Published.xlsx', sheet_name='Lines',
                         engine='openpyxl')
stations_df = pd.read_excel('./Route_choice_Modelling/edited_NBT19_Definitions_Published.xlsx', sheet_name='Stations',
                            engine='openpyxl')

# Reading the definitions data for stops and considering only the underground stations.
stations_df = stations_df.dropna(subset=['MASC'])
stations_df = stations_df[stations_df.MASC.str.endswith('u')]
stations_df = stations_df.reset_index()
stations_df.loc[:, ['stop_id', 'zone_id']] = stations_df.merge(stops_df, left_on='MASC', right_on='MASC',
                                                               how='inner').loc[:, ['stop_id', 'zone_id']]
# Reading the definitions data for lines and considering only the underground lines.
lines_df = lines_df[lines_df.Tsys.str.startswith('u')]
lines_lu = list(lines_df.LineCode)
# lines_lu is a list of all the routes in the london underground
lines_lu.append(None)

# Deleted these routes as they are not underground routes
od_data_2019 = od_data_2019[od_data_2019.route.str.contains('HMSu') == False]
od_data_2019 = od_data_2019[od_data_2019.route.str.contains('HRCu') == False]
# Finding all the routes used between every od pair
od_data_2019.loc[:,
['transfer1_route', 'transfer2_route', 'transfer3_route', 'transfer4_route', 'transfer5_route', 'transfer6_route']] = \
    np.array(od_data_2019['route'].str.split('|', expand=True).loc[:, 0:5])
for i in range(1, 7):
    od_data_2019.loc[:, ['route_{}'.format(i)]] = od_data_2019['transfer{}_route'.format(i)].str[-3:]
print(f"Before filtering the data based on routes present: {len(od_data_2019)}")
# Filter the data based on the route_code
od_data_2019 = od_data_2019[
    od_data_2019[['route_1', 'route_2', 'route_3', 'route_4', 'route_5', 'route_6']].isin(lines_lu).all(axis=1)]
print(f"After filtering the data based on routes present: {len(od_data_2019)}")

# Finding the origin and destination stop ids
lines_df.set_index('LineCode', inplace=True)
lines_dict = lines_df[['LineName']].to_dict(orient='index')
# Removing the Bank station as it is not there in the NUMBAT Description table
stops_df.loc[stops_df.stop_name == 'Bank', 'MASC'] = 'BANu'
stops_df.set_index('MASC', inplace=True)
stops_dict = stops_df[['stop_id', 'zone_id']].to_dict(orient='index')
stop_times_df['arrival_time'] = pd.to_timedelta(stop_times_df['arrival_time'])
trips_df['route_name'] = trips_df.merge(routes_df, left_on='route_id', right_on='route_id', how='left')[
    'route_short_name']
# Finding stops-stops distance matrix
stops_cord = np.array(stops_df[['stop_lat', 'stop_lon']])
stops_stops_matrix = haversine_vector(stops_cord, stops_cord, unit=Unit.METERS, comb=True)
stops_stops_df = pd.DataFrame(stops_stops_matrix, index=stops_df['stop_id'], columns=stops_df['stop_id'])
stops_stops_dict = stops_stops_df.to_dict(orient='index')
stops_trips_dict = {}
for stop in stops_df.stop_id.unique():
    stops_trips_dict[stop] = set(stop_times_df[stop_times_df.stop_id == stop].trip_id.unique())
route_name_trips_dict = {}
for route in routes_df.route_short_name.unique():
    route_name_trips_dict[route] = set(trips_df[trips_df.route_name == route].trip_id.unique())
trips_df.set_index('trip_id', inplace=True)
trips_routes_dict = trips_df[['route_name']].to_dict(orient='index')


def finding_leg_attributes(orig: str, dest: str, route_code: str) -> tuple:
    """
    This function finds the attributes of the route between the origin and destination.
    :param orig: Origin stop name :str
    :param dest: Destination stop name :str
    :return: travel time and fare :tuple
    """
    global stops_dict, stop_times_df, trips_df, fare_rules_df, fare_attributes, avg_fare, route_name_trips_dict, trips_routes_dict
    orig_id, orig_zone = stops_dict[orig]['stop_id'], stops_dict[orig]['zone_id']
    dest_id, dest_zone = stops_dict[dest]['stop_id'], stops_dict[dest]['zone_id']
    data = stop_times_df[stop_times_df.trip_id.isin(
        stops_trips_dict[orig_id] & stops_trips_dict[dest_id] & route_name_trips_dict[route_code])]
    data = data[data.stop_id.isin([orig_id, dest_id])]
    data_org = data.groupby('trip_id').head(1)
    trips = list(data_org[data_org.stop_id == orig_id].trip_id.values)
    try:
        trip_id = trips[0]
        time = data[data.trip_id == trip_id].arrival_time.values
        tt = (pd.to_timedelta(time[1]) - pd.to_timedelta(time[0])).seconds / 60
        route_id = trips_routes_dict[trip_id]['route_name']
        org_idx = data[data.trip_id == trip_id].index.values[0]
        dest_idx = data[data.trip_id == trip_id].index.values[1]
        stop_sequence = list(stop_times_df.loc[org_idx:dest_idx, 'stop_id'].values)
    except IndexError:
        tt = None
        route_id = '0'
        stop_sequence = []
    try:
        fare_id = fare_rules_df[(fare_rules_df['route_id'] == route_id) & (fare_rules_df['origin_id'] == orig_zone) & (
                fare_rules_df['destination_id'] == dest_zone)].fare_id.unique()[0]
        fare = fare_attributes[fare_attributes.fare_id == fare_id].price.values[0]
    except IndexError:
        fare = avg_fare
    return tt, fare, stop_sequence


issue_id = []
issue_od = []
issue_route = []
od_data_2019['fare'] = 0
od_data_2019['Travel_time'] = 0
od_data_2019['stops'] = 0
for id, row in tqdm(od_data_2019.iloc[:, :].iterrows(), total=len(od_data_2019.iloc[:, :])):
    temp = []
    for leg in range(row.legs):
        route = row['transfer{}_route'.format(leg + 1)]
        orig, dest, route_code = re.split(r'>|@', route)
        tt, fare, stop_sequence = finding_leg_attributes(orig, dest, route_code)
        if tt is None:
            issue_od.append((orig, dest))
            issue_id.append(id)
            issue_route.append(route)
            break
        od_data_2019.loc[id, 'Travel_time'] += tt
        od_data_2019.loc[id, 'fare'] += fare
        temp.extend(stop_sequence)
    temp2 = []
    temp2.append(set(pairwise(temp)))
    od_data_2019.loc[id, 'stops'] = [temp2]
# issues_df = pd.DataFrame({'od': issue_od, 'id': issue_id, 'route': issue_route})
# issues_df.to_csv('issues_1.csv')
# Dropping the rows since after rounding off the travellers are very less.
od_data_2019.drop(columns=['1', '8'], inplace=True)
od_data_2019.rename(columns={'od': 'OD', '2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6'}, inplace=True)
od_data_2019_updated = od_data_2019.loc[:, :].loc[~od_data_2019.loc[:, :].index.isin(issue_id), :]
od_data_2019_original_group = pd.DataFrame(od_data_2019.groupby(by=['OD']).size().sort_values(ascending=False))
max_no_of_routes = od_data_2019_original_group.iloc[0, 0]
od_data_2019_updated_grouped = pd.DataFrame(od_data_2019_updated.groupby(by=['OD']).size().sort_values(ascending=False))
od_data_2019_updated_grouped = od_data_2019_updated_grouped.merge(od_data_2019_original_group, left_index=True,
                                                                  right_index=True, how='left')
od_data_2019_updated_grouped = od_data_2019_updated_grouped[
    (od_data_2019_updated_grouped['0_x'] == od_data_2019_updated_grouped['0_y']) & (
            od_data_2019_updated_grouped['0_x'] > 1)]
od_data_2019_updated_grouped = od_data_2019_updated_grouped.reset_index()
od_ls = list(od_data_2019_updated_grouped.OD.unique())
output_2019 = od_data_2019[od_data_2019.OD.isin(od_ls)]


# Finding the overlap distance between routes
def overlap_distance(data):
    """
    This function calculates the overlap distance between the routes.
    :param data: Dataframe containing the routes for od pair.
    :return: tuple containing the array of overlapping distances between the routes and the path size.
    """
    global stops_stops_matrix
    path_size = []
    distance_matrix = np.array([[0] * len(data)] * len(data))
    data.reset_index(inplace=True)
    for id1 in range(len(data)):
        for id2 in range(len(data)):
            common_stops = list(data.loc[id1, 'stops'][0] & data.loc[id2, 'stops'][0])
            distance_matrix[id1, id2] = sum(
                [stops_stops_dict[stop_pair[0]][stop_pair[1]] for stop_pair in common_stops])
    # Calculating the times repeated by route segment in the routes using counter
    combined_stops = []
    for id1 in range(len(data)):
        combined_stops.extend(list(data.loc[id1, 'stops'][0]))
    counter = Counter(combined_stops)
    for id1 in range(len(data)):
        path_size.append(sum(
            stops_stops_dict[stop_pair[0]][stop_pair[1]] / distance_matrix[id1, id1] * np.log(1.0 / counter[stop_pair])
            for stop_pair in data.loc[id1, 'stops'][0]))
    return distance_matrix, path_size


# Initializing the overlap_columns and path size
for i in range(1, max_no_of_routes + 1):
    output_2019['overlap_' + str(i)] = 0
output_2019['path_size'] = 0
# Calculating the overlap distance and path size
for od in tqdm(od_ls):
    no_routes = output_2019[output_2019.OD == od].shape[0]
    distance_matrix, path_size = overlap_distance(output_2019[output_2019.OD == od])
    output_2019.loc[output_2019.OD == od, 'overlap_1':f"overlap_{no_routes}"] = distance_matrix
    output_2019.loc[output_2019.OD == od, 'path_size'] = path_size
output_2019.rename(columns={'tt': 'Travel_time'}, inplace=True)
# Creating the dataframes for different time periods eg: 1 for 7:00 - 10:00
output_2019['fare'] = round(output_2019['fare'] / 80, 2)
output_2019_1 = output_2019.drop(columns=['stops', '2', '3', '4', '5', '6'], axis=1, inplace=False)
output_2019_2 = output_2019.drop(columns=['stops', '1', '3', '4', '5', '6'], axis=1, inplace=False)
output_2019_3 = output_2019.drop(columns=['stops', '2', '1', '4', '5', '6'], axis=1, inplace=False)
output_2019_5 = output_2019.drop(columns=['stops', '2', '3', '4', '1', '6'], axis=1, inplace=False)
output_2019_4 = output_2019.drop(columns=['stops', '2', '3', '1', '5', '6'], axis=1, inplace=False)
output_2019_6 = output_2019.drop(columns=['stops', '2', '3', '4', '5', '1'], axis=1, inplace=False)

# Creating output file for MNL models
out_dict = {'1': output_2019_1, '2': output_2019_2, '3': output_2019_3, '4': output_2019_4, '5': output_2019_5,
            '6': output_2019_6}


def divide_by_alt_num(row, overlap_j, overlap_alt_num):
    return row[overlap_j] / row[overlap_alt_num]


def creating_data_for_mnl_model(output: pd.DataFrame, time_code: str, year: int) -> None:
    """
    This function creates the data for MNL model and saves as file in the current directory
    :param output:
    :param time_code:
    :param year:
    :return:
    """
    global max_no_of_routes
    # Removes the OD if no passenger travelled on that OD in any route
    time_dict = {'2': '7:00:00-10:00:00', '1': '05:00:00-07:00:00', '3': '10:00:00-16:00:00', '4': '16:00:00-19:00:00',
                 '5': '19:00:00-22:00:00', '6': '22:00:00-23:00:00'}
    start_time, end_time = time_dict[time_code].split('-')
    start_hr, start_min, start_sec = [int(x) for x in start_time.split(':')]
    end_hr, end_min, end_sec = [int(x) for x in end_time.split(':')]
    output = output.groupby(by=['OD']).filter(lambda x: x[time_code].sum() > 0)
    # Creates a column with alternative number for each OD
    output['alt_num'] = output.groupby(by=["OD"]).cumcount() + 1
    # Finds the maximum number of alternatives for any OD
    max_no_alt = output['alt_num'].max()
    print(f"Max_no of alternatives: {max_no_alt}")
    output.drop(columns=['route_1', 'route_2', 'route_3', 'route_4', 'route_5', 'route_6'], inplace=True)
    alt_att_2019_dict = {}
    for idx, row in output.iterrows():
        try:
            alt_att_2019_dict[row['OD']][f'tt_{row["alt_num"]}'] = row['Travel_time']
            alt_att_2019_dict[row['OD']][f'fare_{row["alt_num"]}'] = row['fare']
            alt_att_2019_dict[row['OD']][f'avl_{row["alt_num"]}'] = 1
            alt_att_2019_dict[row['OD']][f'transfers_{row["alt_num"]}'] = row['legs']
            alt_att_2019_dict[row['OD']][f'pathsize_{row["alt_num"]}'] = row['path_size']
            alt_att_2019_dict[row['OD']][f'route_{row["alt_num"]}'] = row['route']
        except KeyError:
            alt_att_2019_dict[row['OD']] = {}
            alt_att_2019_dict[row['OD']][f'tt_{row["alt_num"]}'] = row['Travel_time']
            alt_att_2019_dict[row['OD']][f'fare_{row["alt_num"]}'] = row['fare']
            alt_att_2019_dict[row['OD']][f'avl_{row["alt_num"]}'] = 1
            alt_att_2019_dict[row['OD']][f'transfers_{row["alt_num"]}'] = row['legs']
            alt_att_2019_dict[row['OD']][f'pathsize_{row["alt_num"]}'] = row['path_size']
            alt_att_2019_dict[row['OD']][f'route_{row["alt_num"]}'] = row['route']
    alt_att_2019_df = pd.DataFrame.from_dict(alt_att_2019_dict, orient='index')
    alt_att_2019_df = alt_att_2019_df.fillna(0)
    output = output.merge(alt_att_2019_df, left_on='OD', right_index=True, how='left')
    output.drop(columns=['fare', 'Travel_time', 'path_size'], inplace=True)
    # Creating overlap distance ratio of routes with respect to chosen route
    for j in range(1, max_no_of_routes + 1):
        output[f'overlap_percent_{j}'] = output.apply(
            lambda row: divide_by_alt_num(row, f'overlap_{j}', f'overlap_{row["alt_num"]}'), axis=1)
    # Repeating the rows based on the number of passengers choosing the route
    output = output.reindex(output.index.repeat(output[time_code]))
    # Dropping the overlap distance columns
    output.drop(columns=[f'overlap_{j}' for j in range(1, max_no_of_routes + 1)], inplace=True)
    # Resetting the index
    output.reset_index(drop=True, inplace=True)
    # Renaming the alt_num to the choice column
    output.rename(columns={'alt_num': 'choice'}, inplace=True)
    output['departure_time'] = output['choice'].apply(lambda x: datetime.timedelta(
        seconds=random.randint(start_hr * 3600 + start_min * 60 + start_sec, end_hr * 3600 + end_min * 60 + end_sec)))
    # Randomly sampling the data for train and test with 70-30 ratio
    train_df = output.sample(n=int(0.7 * len(output)), random_state=42)
    test_df = output.drop(train_df.index)
    # Saving the data as csv files
    train_df.to_csv(f'./Route_choice_Modelling/Output/train_{year}_{time_code}.csv', index=True, index_label='Id')
    test_df.to_csv(f'./Route_choice_Modelling/Output/test_{year}_{time_code}.csv', index=True, index_label='Id')
    return None


# creating_data_for_mnl_model(out_dict['1'], '1', 2019)
for time_interval in out_dict.keys():
    print(f"Time_interval: {time_interval}")
    creating_data_for_mnl_model(out_dict[time_interval], time_interval, 2019)
# output_2017 = pd.read_csv('output_2017.csv')
