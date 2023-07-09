import datetime
import pandas as pd
from zipfile import ZipFile
import time
import numpy as np
from haversine import haversine_vector, Unit
from itertools import tee
import builtins
import warnings
from collections import Counter
import random
from multiprocessing import Pool
from tqdm import tqdm

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
    """
    Reads GTFS data
    :param filename: str,
    :return: None
    """
    # Reading GTFS data
    global routes_df, trips_df, stop_times_df, stops_df
    with ZipFile(filename) as zip:
        with zip.open('London Underground GTFS/routes.txt') as routes:
            routes_df = pd.read_csv(routes)
        with zip.open('London Underground GTFS/trips.txt') as trips:
            trips_df = pd.read_csv(trips)
        with zip.open('London Underground GTFS/stop_times.txt') as stop_times:
            stop_times_df = pd.read_csv(stop_times)
        with zip.open('London Underground GTFS/stops.txt') as stops:
            stops_df = pd.read_csv(stops)
    return None


def preProcessing(gtfs_path: str, file_path: str, year: int) -> None:
    """
    Preprocessing the data for the OD data and creates global variables od_data, routes_df, trips_df, stop_times_df
    :param gtfs_path: str, path to the GTFS data
    :param file_path: str, path to the file containing the OD data
    :param year: int, year of the OD data
    :return: None
    """
    global od_data, routes_df, trips_df, stop_times_df, stops_df
    read_GTFS_data(gtfs_path)
    if year == 2017:
        # file_path = './Route_choice_Modelling/Route choice by origin-destination pair 2017.xls'
        od_data = pd.read_excel(file_path, skiprows=3)
        od_data.columns = ['origin_id', 'origin_name', 'destination_id', 'destination_name', 'transfer1', 'transfer2',
                           'transfer3', 'transfer4', '1', '2', '3', '4', '5', '6', 'total']

        # Removing the spaces at the end of the name.
        columns_ls = ['origin_name', 'destination_name', 'transfer1', 'transfer2', 'transfer3', 'transfer4']
        for stops_ls in columns_ls:
            od_data[stops_ls] = list(od_data[stops_ls].apply(lambda x: x.rstrip()))

        # Changing the name of the stations
        rename_dict = {"St. John's Wood": "St John's Wood",
                       'Paddington London': 'Paddington', 'Hammersmith (H&C)': 'Hammersmith',
                       'Shepherds Bush': "Shepherd's Bush",
                       'Hammersmith (Dis)': 'Hammersmith', "King's Cross St. Pancras": "King's Cross St Pancras",
                       "St. James's Park": "St James's Park", "St. Paul's": "St Paul's", 'Bank / Monument': 'Bank',
                       "Shepherd's Bush (Cen)": "Shepherd's Bush Market",
                       'Waterloo London Br': 'Waterloo', 'Euston Br': 'Euston', 'Marylebone London': 'Marylebone',
                       'Liverpool Street (London) Br': 'Liverpool Street',
                       'West Hampstead Thameslink': 'West Hampstead'}
        od_data = od_data.replace(rename_dict)
        # Filtering the data
        stops = list(stops_df.stop_name.unique())
        stops_df.set_index('stop_name', inplace=True)
        stops_dict = stops_df[['stop_id']].to_dict(orient='index')
        # Checking origin and destination are in the LU stops
        od_data = od_data[od_data.origin_name.isin(stops) & od_data.destination_name.isin(stops)]
        # Finding the no of legs
        od_data['route'] = od_data['origin_name'].apply(lambda x: [stops_dict[x]['stop_id']])
        for j in range(1, 5):
            od_data = od_data[
                (od_data[f'transfer{j}'] == '') | (od_data[f'transfer{j}'].isin(stops))]
            print(f"After removing checking the stops in transfer{j}, no of routes remaining are {len(od_data)}")
            od_data.loc[od_data[f'transfer{j}'] != '', ['route', f'transfer{j}']].apply(
                lambda x: x.route.append(stops_dict[x[f'transfer{j}']]['stop_id']), axis=1)
        od_data.loc[:, ['route', 'destination_name']].apply(
            lambda x: x.route.append(stops_dict[x.destination_name]['stop_id']), axis=1)
        od_data['transfers'] = od_data['route'].apply(lambda x: len(x) - 1)
        od_data['od'] = od_data['origin_name'] + ' - ' + od_data['destination_name']
    elif year == 2019:
        # file_path = './Route_choice_Modelling/NBT19MTT3a_routechoice_tb_wf.csv'
        # Reading 2019 od data
        od_data = pd.read_csv(file_path)
        lines_df = pd.read_excel('./edited_NBT19_Definitions_Published.xlsx', sheet_name='Lines', engine='openpyxl')
        # Reading the definitions data for lines and considering only the underground lines.
        lines_df = lines_df[lines_df.Tsys.str.startswith('u')]
        lines_lu = list(lines_df.LineCode)
        # lines_lu is a list of all the routes in the london underground
        lines_lu.append(None)
        # Deleted these routes as they are not underground routes
        od_data = od_data[od_data.route.str.contains('HMSu') == False]
        od_data = od_data[od_data.route.str.contains('HRCu') == False]
        # Finding all the routes used between every od pair
        od_data.loc[:,
        ['transfer1_route', 'transfer2_route', 'transfer3_route', 'transfer4_route', 'transfer5_route',
         'transfer6_route']] = \
            np.array(od_data['route'].str.split('|', expand=True).loc[:, 0:5])
        for i in range(1, 7):
            od_data.loc[:, ['route_{}'.format(i)]] = od_data['transfer{}_route'.format(i)].str[-3:]
        print(f"Before filtering the data based on routes present: {len(od_data)}")
        # Filter the data based on the route_code
        od_data = od_data[
            od_data[['route_1', 'route_2', 'route_3', 'route_4', 'route_5', 'route_6']].isin(lines_lu).all(axis=1)]
        print(f"After filtering the data based on routes present: {len(od_data)}")
        # Finding the no of legs
        od_data['transfers'] = od_data['legs'] - 1
        # Making route as list of stop_ids
        # Changing the bank MASC code such that monument is not considered as bank
        stops_df.loc[stops_df.stop_name == 'Bank', 'MASC'] = 'BANu'
        stops_df.set_index('MASC', inplace=True)
        stops_dict = stops_df[['stop_id']].to_dict(orient='index')
        od_data['route'] = od_data['legs'].apply(lambda x: [])
        for i in range(1, 7):
            if i != 1:
                od_data.loc[od_data[f'transfer{i}_route'].notnull(), ['route', f'transfer{i}_route']].apply(
                    lambda x: x.route.append(stops_dict[x[f'transfer{i}_route'][5:9]]['stop_id']), axis=1)
            else:
                od_data.loc[od_data[f'transfer{i}_route'].notnull(), ['route', f'transfer{i}_route']].apply(
                    lambda x: x.route.extend([stops_dict[x[f'transfer{i}_route'][:4]]['stop_id'],
                                              stops_dict[x[f'transfer{i}_route'][5:9]]['stop_id']]), axis=1)
        # Dropping 1 and 8 columns
        od_data.drop(columns=['1', '8', 'route_1'], inplace=True)
        # Changing the column names
        od_data.rename(columns={'2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6'}, inplace=True)
        # Rounding off the values in the dataframe
        od_data[['1', '2', '3', '4', '5', '6']] = od_data[['1', '2', '3', '4', '5', '6']].round(0)
    # Considering the only OD's with more than 1 routes
    od_data_grouped = od_data.groupby('od').size().sort_values(ascending=True)
    od_data = od_data[od_data.od.isin(od_data_grouped[od_data_grouped > 1].index)]
    print(f"After preprocessing the data: {len(od_data)}")
    # Considering only the required columns
    od_data = od_data[['od', 'route', 'transfers', '1', '2', '3', '4', '5', '6']]
    return None


def prepIndividualDataset():
    global od_data, max_alt
    # Creating available routes for each OD pair
    # Creating a column with route number for the respective OD pair
    od_data['choice'] = od_data.groupby('od').cumcount() + 1
    max_alt = od_data.choice.max()

    # Creating the dataframe with OD as index and their respective routes as columns
    alternative_routes = od_data.pivot(index='od', columns='choice', values='route')
    alternative_routes.columns = ['route_{}'.format(i) for i in range(1, len(alternative_routes.columns) + 1)]

    # Merging the dataframe with the original dataframe
    od_data = od_data.merge(alternative_routes, on='od')

    # Filling Nan with zero in alternative routes
    od_data.loc[:, alternative_routes.columns] = od_data.loc[:, alternative_routes.columns].fillna(0)
    return None


def stops_trips_fn():
    global stops_trips_dict, stops_df, stop_times_df, leg_stops_sequence,route_name_trips_dict
    stops_trips_dict = {}
    stop_times_df['arrival_time'] = pd.to_timedelta(stop_times_df['arrival_time'])
    for stop in stops_df.stop_id.unique():
        stops_trips_dict[stop] = set(stop_times_df[stop_times_df.stop_id == stop].trip_id.unique())
    stop_times_df.loc[(stop_times_df['arrival_time'] >= pd.to_timedelta('00:00:00')) & (
            stop_times_df['arrival_time'] < pd.to_timedelta('02:00:00')), 'arrival_time'] += pd.to_timedelta('1 days')
    leg_stops_sequence = {}
    route_name_trips_dict = {}
    for route in routes_df.route_short_name.unique():
        route_name_trips_dict[route] = set(trips_df[trips_df.route_name == route].trip_id.unique())
    return None


def leg_wise_attributes(orig_id: str, dest_id: str, route:str, departure_time: str) -> tuple:
    """
    This function estimates the leg wise attributes for the given origin and destination using the arrival time at the
    origin of leg
    :param orig_id: str, origin stop_id of the leg
    :param dest_id: str, destination stop_id of the leg
    :param departure_time: str, departure time from the origin stop_id
    :return: tuple, (IVTT,wait_time,stop_sequence,arrival_time)
    """
    global stop_times_df, stops_df, route_name_trips_dict, stops_trips_dict, leg_stops_sequence,route_names_trips_dict
    # Finding the trips that are common between the origin and destination
    trips = list(route_name_trips_dict[route] & stops_trips_dict[orig_id] & stops_trips_dict[dest_id])
    data = stop_times_df[stop_times_df.trip_id.isin(trips)]
    # Finding the rows with the origin and destination
    data = data[data.stop_id.isin([orig_id, dest_id])]
    # Sorting the data based on arrival_time
    data.sort_values(by=['arrival_time'], inplace=True)
    # Finding the trips which are going to origin_id to destination_id
    data_org = data.groupby('trip_id').head(1)
    data_org = data_org[data_org.stop_id == orig_id]
    # Filtering the trips which are arriving after the departure time
    data_org = data_org[data_org.arrival_time >= departure_time]
    trips = list(data_org.trip_id.values)
    try:
        trip_id = trips[0]
        wait_time = (pd.to_timedelta(data_org[data_org.trip_id == trip_id].arrival_time.values[0]) - pd.to_timedelta(
            departure_time)).seconds / 60
        dept_time = pd.to_timedelta(data[data.trip_id == trip_id].arrival_time.values[1])
        IVTT = (pd.to_timedelta(dept_time) - pd.to_timedelta(departure_time)).seconds / 60 - wait_time
        try:
            stop_sequence = leg_stops_sequence[(orig_id, dest_id)]
        except KeyError:
            org_idx = data[data.trip_id == trip_id].index.values[0]
            dest_idx = data[data.trip_id == trip_id].index.values[1]
            stop_sequence = list(stop_times_df.loc[org_idx:dest_idx, 'stop_id'].values)
            leg_stops_sequence[(orig_id, dest_id)] = stop_sequence
    except IndexError:
        wait_time, dept_time, IVTT, stop_sequence = -10000, departure_time, -10000, []
    return IVTT, wait_time, stop_sequence, dept_time


def route_wise_attributes(route: list, dept_time: pd.to_timedelta) -> dict:
    """
    This function calculates the route wise attributes by iterating over the route for each leg using the
    leg_wise_attributes function.
    :param route: list of stops only origin, transfer and destination
    :param dept_time: pd.to_timedelta, arrival time at the origin
    :return: dict with route wise attributes
    """
    global stop_times_df, stops_df, route_name_trips_dict, stops_trips_dict
    route_attributes = {'IVTT': 0, 'stop_sequence': [], 'initial_wait_time': 0, 'transfer_wait_time': 0, 'Not_found': 0}
    for i in range(len(route) - 1):
        IVTT, wait_time, stop_sequence, dept_time = leg_wise_attributes(route[i], route[i + 1],route,dept_time)
        if IVTT == -10000:
            route_attributes['Not_found'] = 1
            break
        else:
            if i == 0:
                route_attributes['initial_wait_time'] = wait_time
                route_attributes['IVTT'] += IVTT
                route_attributes['stop_sequence'].extend(stop_sequence)
            else:
                route_attributes['transfer_wait_time'] += wait_time
                route_attributes['IVTT'] += IVTT
                route_attributes['stop_sequence'].extend(stop_sequence)
    route_attributes['total_travel_time'] = route_attributes['initial_wait_time'] + route_attributes[
        'transfer_wait_time'] + route_attributes['IVTT']
    route_attributes['stop_sequence'] = set(pairwise(route_attributes['stop_sequence']))
    return route_attributes


def row_wise_attributes(row: dict) -> dict:
    """
    This function calculates the attributes of the row i.e., individual commute for every available alternative
    using route_wise attributes function.
    :param row: dictionary of the row of the dataframe
    :return: dictionary of the attributes of the row
    """
    global max_alt
    row_attributes = {'Not_found': 0}
    for alt in range(1, max_alt + 1):
        route = row[f'route_{alt}']
        if route == 0:
            row_attributes[f'transfer_{alt}'] = 0
            row_attributes[f'IVTT_{alt}'] = 0
            row_attributes[f'stop_sequence_{alt}'] = set()
            row_attributes[f'initial_wait_time_{alt}'] = 0
            row_attributes[f'transfer_wait_time_{alt}'] = 0
            row_attributes[f'total_travel_time_{alt}'] = 0
            row_attributes[f'walking_time_{alt}'] = 0
            row_attributes[f'avl_{alt}'] = 0
        else:
            route_attributes = route_wise_attributes(route, row['departure_time'])
            row_attributes[f'transfer_{alt}'] = len(route) - 2
            if route_attributes['Not_found'] == 1:
                row_attributes[f'IVTT_{alt}'] = 0
                row_attributes[f'stop_sequence_{alt}'] = set()
                row_attributes[f'initial_wait_time_{alt}'] = 0
                row_attributes[f'transfer_wait_time_{alt}'] = 0
                row_attributes[f'total_travel_time_{alt}'] = 0
                row_attributes[f'walking_time_{alt}'] = 0
                row_attributes[f'avl_{alt}'] = 1
                row_attributes[f'Not_found'] = 1
            else:
                row_attributes[f'IVTT_{alt}'] = route_attributes['IVTT']
                row_attributes[f'stop_sequence_{alt}'] = route_attributes['stop_sequence']
                row_attributes[f'initial_wait_time_{alt}'] = route_attributes['initial_wait_time']
                row_attributes[f'transfer_wait_time_{alt}'] = route_attributes['transfer_wait_time']
                row_attributes[f'total_travel_time_{alt}'] = route_attributes['total_travel_time']
                row_attributes[f'walking_time_{alt}'] = 0
                row_attributes[f'avl_{alt}'] = 1
    return row_attributes


def create_individual_time_interval() -> None:
    """
    This function creates individual level data points for each time interval and stores in global dictionary with
    keys as 1,2,3,4,5,6 and values as dataframes
    :return: None
    """
    global od_data, od_timeInterval_dict
    od_timeInterval_dict = {}
    time_dict = {'2': '7:00:00-10:00:00', '1': '05:00:00-07:00:00', '3': '10:00:00-16:00:00', '4': '16:00:00-19:00:00',
                 '5': '19:00:00-22:00:00', '6': '22:00:00-23:00:00'}
    od_timeInterval_dict['1'] = od_data.drop(columns=['2', '3', '4', '5', '6'], axis=1, inplace=False)
    od_timeInterval_dict['2'] = od_data.drop(columns=['1', '3', '4', '5', '6'], axis=1, inplace=False)
    od_timeInterval_dict['3'] = od_data.drop(columns=['2', '1', '4', '5', '6'], axis=1, inplace=False)
    od_timeInterval_dict['4'] = od_data.drop(columns=['2', '3', '1', '5', '6'], axis=1, inplace=False)
    od_timeInterval_dict['5'] = od_data.drop(columns=['2', '3', '4', '1', '6'], axis=1, inplace=False)
    od_timeInterval_dict['6'] = od_data.drop(columns=['2', '3', '4', '5', '1'], axis=1, inplace=False)
    for key_t in od_timeInterval_dict.keys():
        # Creating each row as individual based on the time interval column
        od_timeInterval_dict[key_t] = od_timeInterval_dict[key_t].reindex(
            od_timeInterval_dict[key_t].index.repeat(od_timeInterval_dict[key_t][key_t]))
        # Creating departure time column
        start_time, end_time = time_dict[key_t].split('-')
        start_hr, start_min, start_sec = [int(x) for x in start_time.split(':')]
        end_hr, end_min, end_sec = [int(x) for x in end_time.split(':')]
        od_timeInterval_dict[key_t]['departure_time'] = od_timeInterval_dict[key_t]['choice'].apply(
            lambda x: datetime.timedelta(seconds=random.randint(start_hr * 3600 + start_min * 60 + start_sec,
                                                                end_hr * 3600 + end_min * 60 + end_sec)))
        od_timeInterval_dict[key_t].reset_index(inplace=True, drop=True)
    return None


def haversine_distance() -> None:
    """
    This function calculates the distance between each stop pair using haversine formula and stores in global variable
    :return: None
    """
    global stops_stops_dict, stops_data, ps_od_dict
    # Creating the array of stops_coordinates
    stops_cord = np.array(stops_df[['stop_lat', 'stop_lon']])

    # Creating the stops-stops distance matrix using haversine
    stops_stops_matrix = haversine_vector(stops_cord, stops_cord, unit=Unit.METERS, comb=True)

    # Converting the matrix to dataframe and dictionary for easy access
    stops_stops_df = pd.DataFrame(stops_stops_matrix, index=stops_df['stop_id'], columns=stops_df['stop_id'])
    stops_stops_dict = stops_stops_df.to_dict(orient='index')

    # Creating the dictionary of path size for each od pair
    ps_od_dict = {}
    return None


def path_size(row):
    """
    This function calculates the path size for each alternative
    :param row: dictionary of
    :return:
    """
    global stops_stops_dict, max_alt, ps_od_dict
    path_size_dict = ps_od_dict.get(row['od'], {})
    if path_size_dict == {}:
        combined_stops = []
        for alt in range(1, max_alt + 1):
            if row[f'avl_{alt}'] == 1:
                combined_stops.extend(list(row[f'stop_sequence_{alt}']))
        counter = Counter(combined_stops)
        for alt in range(1, max_alt + 1):
            distance = 0
            ps = 0
            for stop_stop in row[f'stop_sequence_{alt}']:
                distance += stops_stops_dict[stop_stop[0]][stop_stop[1]]
                ps += stops_stops_dict[stop_stop[0]][stop_stop[1]] * np.log(1 / counter[stop_stop])
            try:
                ps = ps / distance
            except ZeroDivisionError:
                ps = 0
            path_size_dict[f'path_size_{alt}'] = ps
        ps_od_dict[row['od']] = path_size_dict
    return path_size_dict


if __name__ == '__main__':
    year = 2019
    # preProcessing(gtfs_path='GTFS.zip',
    #               file_path='NBT19MTT3a_routechoice_tb_wf.csv', year=year)
    preProcessing(gtfs_path='./Route_choice_Modelling/GTFS.zip',
                  file_path='./Route_choice_Modelling/Route choice by origin-destination pair 2017.xls', year=year)
    prepIndividualDataset()
    stops_trips_fn()
    create_individual_time_interval()
    haversine_distance()
    print(f'Preprocessing done at {datetime.datetime.now().strftime("%H:%M:%S")}')
    print('-' * 150)
    for key in od_timeInterval_dict.keys():
        print(f'Processing for time interval {key} and its length is {len(od_timeInterval_dict[key])}')
        # Converting dataframe to a list of dictionaries where each dictionary is a row.
        od_timeInterval_df = od_timeInterval_dict[key].loc[:]
        route_columns = [f'route_{i}' for i in range(1, max_alt + 1)]
        od_timeInterval_df.loc[:, route_columns] = od_timeInterval_df.loc[:, route_columns].fillna(0)
        od_timeInterval = od_timeInterval_df.to_dict('records')

        # Multiprocessing
        CORES = 20
        start_time = time.time()
        # Times and transfer attributes
        with Pool(CORES) as pool:
            results = pool.map(row_wise_attributes, od_timeInterval)
        results_df = pd.DataFrame(results)
        # Concatenate the attributes derived from the function to the original dataframe
        od_timeInterval_df = pd.concat([od_timeInterval_df, results_df], axis=1)

        # path size attributes
        od_timeInterval = od_timeInterval_df.to_dict('records')
        od_timeInterval = od_timeInterval_df.to_dict('records')
        with Pool(CORES) as pool:
            results = pool.map(path_size, od_timeInterval)
        results_df = pd.DataFrame(results)
        # Concatenate the attributes derived from the function to the original dataframe
        od_timeInterval_df = pd.concat([od_timeInterval_df, results_df], axis=1)

        # Saving the dataframe
        print(f'Time taken for the parallel with 20 cores: {(time.time() - start_time)}')
        od_timeInterval_df.to_csv(f'./Output/od_{year}_{key}.csv', index=False)
        print(f'File saved in ./Route_choice_Modelling/Output/od_{year}_{key}.csv')
        print(f'This interval done at {datetime.datetime.now().strftime("%H:%M:%S")}')
        print('-' * 150)
