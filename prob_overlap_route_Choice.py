import datetime
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
def pairwise(iterable: list) -> zip:
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    :param iterable: list
    :return: zip
    """
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
avg_fare = fare_attributes['price'].mean()

# Read the data 2017
od_data_2017 = pd.read_excel('./Route_choice_Modelling/Route choice by origin-destination pair 2017.xls', skiprows=3)
od_data_2017.columns = ['origin_id', 'origin_name', 'destination_id', 'destination_name', 'transfer1', 'transfer2',
                        'transfer3', 'transfer4', '1', '2', '3', '4', '5', '6', 'total']

# Removing the spaces at the end of the name.
columns_ls = ['origin_name', 'destination_name', 'transfer1', 'transfer2', 'transfer3', 'transfer4']
for stops_ls in columns_ls:
    od_data_2017[stops_ls] = list(od_data_2017[stops_ls].apply(lambda x: x.rstrip()))

# Changing the name of the stations
rename_dict = {"St. John's Wood": "St John's Wood",
               'Paddington London': 'Paddington', 'Hammersmith (H&C)': 'Hammersmith',
               'Shepherds Bush': "Shepherd's Bush",
               'Hammersmith (Dis)': 'Hammersmith', "King's Cross St. Pancras": "King's Cross St Pancras",
               "St. James's Park": "St James's Park", "St. Paul's": "St Paul's", 'Bank / Monument': 'Bank',
               "Shepherd's Bush (Cen)": "Shepherd's Bush Market",
               'Waterloo London Br': 'Waterloo', 'Euston Br': 'Euston', 'Marylebone London': 'Marylebone',
               'Liverpool Street (London) Br': 'Liverpool Street', 'West Hampstead Thameslink': 'West Hampstead'}
od_data_2017 = od_data_2017.replace(rename_dict)
# Finding the no of legs
od_data_2017['No_of_legs'] = 0
for j in range(1, 5):
    od_data_2017.loc[(od_data_2017[f'transfer{j}'] != '') & (od_data_2017.No_of_legs == 0), 'No_of_legs'] = j

# Finding stops-stops distance matrix
# Creating the array of stops_coordinates
stops_cord = np.array(stops_df[['stop_lat', 'stop_lon']])
# Creating the stops-stops distance matrix using haversine
stops_stops_matrix = haversine_vector(stops_cord, stops_cord, unit=Unit.METERS, comb=True)
# Converting the matrix to dataframe and dictionary for easy access
stops_stops_df = pd.DataFrame(stops_stops_matrix, index=stops_df['stop_id'], columns=stops_df['stop_id'])
stops_stops_dict = stops_stops_df.to_dict(orient='index')
# Filtering the data
stops_df.set_index('stop_name', inplace=True)
stops = list(stops_df.index)
# Checking origin and destination are in the LU stops
od_data_2017 = od_data_2017[od_data_2017.origin_name.isin(stops) & od_data_2017.destination_name.isin(stops)]
# Checking transfer stops are in the LU stops
for j in range(1, 5):
    od_data_2017 = od_data_2017[(od_data_2017[f'transfer{j}'] == '') | (od_data_2017[f'transfer{j}'].isin(stops))]
    print(f"After removing checking the stops in transfer{j}, no of routes remaining are {len(od_data_2017)}")
# Making a dictionary of stops which stores zone_id and stop_id
stops_dict = stops_df[['stop_id', 'zone_id']].to_dict(orient='index')
# Converting the arrival time to timedelta
stop_times_df['arrival_time'] = pd.to_timedelta(stop_times_df['arrival_time'])
# Adding route_short_name column to trips_df
trips_df['route_name'] = trips_df.merge(routes_df, left_on='route_id', right_on='route_id', how='left')[
    'route_short_name']
# Finding the trips for each stop
stops_trips_dict = {}
for stop in stops:
    stops_trips_dict[stop] = set(stop_times_df[stop_times_df.stop_id == stops_dict[stop]['stop_id']].trip_id.unique())


def finding_leg_attributes(orig: str, dest: str) -> tuple:
    """
    This function finds the attributes of the route between the origin and destination.
    :param orig: Origin stop name :str
    :param dest: Destination stop name :str
    :return: travel time,fare,stop_sequence and route_id :tuple
    """
    global stops_dict, stop_times_df, trips_df, fare_rules_df, fare_attributes, avg_fare
    orig_id, orig_zone = stops_dict[orig]['stop_id'], stops_dict[orig]['zone_id']
    dest_id, dest_zone = stops_dict[dest]['stop_id'], stops_dict[dest]['zone_id']
    data = stop_times_df[
        stop_times_df.trip_id.isin(stops_trips_dict[orig] & stops_trips_dict[dest]) & stop_times_df.stop_id.isin(
            [orig_id, dest_id])]
    data_org = data.groupby('trip_id').head(1)
    trips = list(data_org[data_org.stop_id == orig_id].trip_id.values)
    try:
        trip_id = trips[0]
        time = data[data.trip_id == trip_id].arrival_time.values
        tt = (pd.to_timedelta(time[1]) - pd.to_timedelta(time[0])).seconds / 60
        route_name = trips_df[trips_df.trip_id == trip_id].route_id.values[0]
        route_id = routes_df[routes_df.route_id == route_name].route_short_name.values[0]
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
    return tt, fare, stop_sequence, route_id


# Finding the travel time and fare for each route
issues = []
issues_id = []
od_data_2017['Travel_time'] = 0
od_data_2017['fare'] = 0
od_data_2017['stops'] = 0
od_data_2017['route'] = 0
# travel_time_ls = []
# fare_ls = []
# route_ls = []
od_data_2017.reset_index(drop=True, inplace=True)
for id, row in tqdm(od_data_2017.iloc[:, :].iterrows(), total=len(od_data_2017.iloc[:, :])):
    org = 1
    temp = []
    temp3 = []
    for transfer in range(1, row.No_of_legs + 2):
        orig = row[org]
        if transfer == row.No_of_legs + 1:
            dest = row[3]
        else:
            dest = row[3 + transfer]
        tt, fare, stops_sequence, route_id = finding_leg_attributes(orig, dest)
        if tt is None:
            issues.append((orig, dest))
            issues_id.append(id)
            break
        org = 3 + transfer
        od_data_2017.loc[id, 'Travel_time'] += tt
        od_data_2017.loc[id, 'fare'] += fare
        t = [orig, dest, route_id]
        temp3.extend([t])
        temp.extend(stops_sequence)
    temp2 = []
    temp2.append(set(pairwise(temp)))
    temp4 = [temp3]
    od_data_2017.loc[id, 'stops'] = [temp2]
    od_data_2017.loc[id, 'route'] = [temp4]

# Removing the rows with no route found
od_data_2017_updated = od_data_2017.loc[:, :].loc[~od_data_2017.loc[:, :].index.isin(issues_id), :]
od_data_2017_original_group = pd.DataFrame(
    od_data_2017.groupby(by=['origin_name', 'destination_name']).size().sort_values(ascending=False))
od_data_2017_updated_grouped = pd.DataFrame(
    od_data_2017_updated.groupby(by=['origin_name', 'destination_name']).size().sort_values(ascending=False))
max_no_of_routes = od_data_2017_original_group.iloc[0, 0]
od_data_2017_updated_grouped = od_data_2017_updated_grouped.merge(od_data_2017_original_group, left_index=True,
                                                                  right_index=True, how='left')
# Removing the 'OD' if the number of routes are decreasing with respect to original data.
od_data_2017_updated_grouped = od_data_2017_updated_grouped[
    (od_data_2017_updated_grouped['0_x'] == od_data_2017_updated_grouped['0_y']) & (
            od_data_2017_updated_grouped['0_x'] > 1)]
od_data_2017_updated_grouped = od_data_2017_updated_grouped.reset_index()
od_ls = list(od_data_2017_updated_grouped.origin_name + ' - ' + od_data_2017_updated_grouped.destination_name)
od_data_2017['OD'] = od_data_2017['origin_name'] + ' - ' + od_data_2017['destination_name']
output_2017 = od_data_2017[od_data_2017.OD.isin(od_ls)]
output_2017.to_csv('output_2017.csv', index=False)


# Finding the overlap distance between routes
def overlap_distance(data):
    """"
    This function calculates the overlap distance between routes.
    :param data: The dataframe containing the routes
    :return: The overlap distance between routes
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
    output_2017['overlap_' + str(i)] = 0
output_2017['path_size'] = 0
# Calculating the overlap distance and path size
for od in tqdm(od_ls):
    no_routes = output_2017[output_2017.OD == od].shape[0]
    distance_matrix, path_size = overlap_distance(output_2017[output_2017.OD == od])
    output_2017.loc[output_2017.OD == od, 'overlap_1':f"overlap_{no_routes}"] = distance_matrix
    output_2017.loc[output_2017.OD == od, 'path_size'] = path_size
output_2017.rename(columns={'tt': 'Travel_time'}, inplace=True)
# Creating the dataframes for different time periods eg: 1 for 7:00 - 10:00
output_2017['fare'] = round(output_2017['fare'] / 80, 2)
output_2017_1 = output_2017.drop(columns=['stops', '2', '3', '4', '5', '6'], axis=1, inplace=False)
output_2017_2 = output_2017.drop(columns=['stops', '1', '3', '4', '5', '6'], axis=1, inplace=False)
output_2017_3 = output_2017.drop(columns=['stops', '2', '1', '4', '5', '6'], axis=1, inplace=False)
output_2017_4 = output_2017.drop(columns=['stops', '2', '3', '1', '5', '6'], axis=1, inplace=False)
output_2017_5 = output_2017.drop(columns=['stops', '2', '3', '4', '1', '6'], axis=1, inplace=False)
output_2017_6 = output_2017.drop(columns=['stops', '2', '3', '4', '5', '1'], axis=1, inplace=False)

# Creating output file for MNL models
out_dict = {'1': output_2017_1, '2': output_2017_2, '3': output_2017_3, '4': output_2017_4, '5': output_2017_5,
            '6': output_2017_6}


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
    # Removes the OD if no passenger travelled on that OD in any route
    time_dict = {'2': '7:00:00-10:00:00', '1': '04:38:00-07:00:00', '3': '10:00:00-16:00:00', '4': '16:00:00-19:00:00',
                 '5': '19:00:00-22:00:00', '6': '22:00:00-23:00:00'}
    start_time, end_time = time_dict[time_code].split('-')
    start_hr, start_min, start_sec = [int(x) for x in start_time.split(':')]
    end_hr, end_min, end_sec = [int(x) for x in end_time.split(':')]
    output = output.groupby(by=['origin_name', 'destination_name']).filter(
        lambda x: x[time_code].sum() > 0)
    # Creates a column with alternative number for each OD
    output['alt_num'] = output.groupby(by=['origin_name', 'destination_name']).cumcount() + 1
    output.loc[:, 'OD'] = output['origin_name'] + ' - ' + output['destination_name']
    # Finds the maximum number of alternatives for any OD
    max_no_alt = output['alt_num'].max()
    print(f"Max_no of alternatives: {max_no_alt}")
    alt_att_2017_dict = {}
    for idx, row in output.iterrows():
        try:
            alt_att_2017_dict[row['OD']][f'tt_{row["alt_num"]}'] = row['Travel_time']
            alt_att_2017_dict[row['OD']][f'fare_{row["alt_num"]}'] = row['fare']
            alt_att_2017_dict[row['OD']][f'avl_{row["alt_num"]}'] = 1
            alt_att_2017_dict[row['OD']][f'transfers_{row["alt_num"]}'] = row['No_of_legs']
            alt_att_2017_dict[row['OD']][f'pathsize_{row["alt_num"]}'] = row['path_size']
            alt_att_2017_dict[row['OD']][f'route_{row["alt_num"]}'] = row['route'][0]
        except KeyError:
            alt_att_2017_dict[row['OD']] = {}
            alt_att_2017_dict[row['OD']][f'tt_{row["alt_num"]}'] = row['Travel_time']
            alt_att_2017_dict[row['OD']][f'fare_{row["alt_num"]}'] = row['fare']
            alt_att_2017_dict[row['OD']][f'avl_{row["alt_num"]}'] = 1
            alt_att_2017_dict[row['OD']][f'transfers_{row["alt_num"]}'] = row['No_of_legs']
            alt_att_2017_dict[row['OD']][f'pathsize_{row["alt_num"]}'] = row['path_size']
            alt_att_2017_dict[row['OD']][f'route_{row["alt_num"]}'] = row['route'][0]
    alt_att_2017_df = pd.DataFrame.from_dict(alt_att_2017_dict, orient='index')
    alt_att_2017_df = alt_att_2017_df.fillna(0)
    output = output.merge(alt_att_2017_df, left_on='OD', right_index=True, how='left')
    output.drop(
        columns=['origin_id', 'destination_id', 'fare', 'Travel_time', 'No_of_legs', 'path_size'], inplace=True)
    # Creating overlap distance ratio of routes with respect to chosen route
    for j in range(1, 14):
        output[f'overlap_percent_{j}'] = output.apply(
            lambda row: divide_by_alt_num(row, f'overlap_{j}', f'overlap_{row["alt_num"]}'), axis=1)
    # Repeating the rows based on the number of passengers choosing the route
    output = output.reindex(output.index.repeat(output[time_code]))
    # Dropping the overlap distance columns
    output.drop(columns=[f'overlap_{j}' for j in range(1, 14)], inplace=True)
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


for time_interval in out_dict.keys():
    print(f"Time_interval: {time_interval}")
    creating_data_for_mnl_model(out_dict[time_interval], time_interval, 2017)
# output_2017 = pd.read_csv('output_2017.csv')
