import re
import pandas as pd
from zipfile import ZipFile
from multiprocessing import Pool
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")


def read_GTFS_data(filename: str) -> None:
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


def Reading_od_data(file_name: str):
    global od_data_2019, routes_df, trips_df, stop_times_df, stops_df, stops_trips_dict, lines_dict, stops_dict, route_name_trips_dict
    # Reading 2019 od data
    od_data_2019 = pd.read_csv(file_name)
    lines_df = pd.read_excel('./Route_choice_Modelling/edited_NBT19_Definitions_Published.xlsx', sheet_name='Lines',
                             engine='openpyxl')

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
    ['transfer1_route', 'transfer2_route', 'transfer3_route', 'transfer4_route', 'transfer5_route',
     'transfer6_route']] = \
        np.array(od_data_2019['route'].str.split('|', expand=True).loc[:, 0:5])
    #
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

    stops_trips_dict = {}
    for stop in stops_df.stop_id.unique():
        stops_trips_dict[stop] = set(stop_times_df[stop_times_df.stop_id == stop].trip_id.unique())
    route_name_trips_dict = {}
    for route in routes_df.route_short_name.unique():
        route_name_trips_dict[route] = set(trips_df[trips_df.route_name == route].trip_id.unique())
    trips_df.set_index('trip_id', inplace=True)
    trips_routes_dict = trips_df[['route_name']].to_dict(orient='index')
    stop_times_df.loc[(stop_times_df['arrival_time'] >= pd.to_timedelta('00:00:00')) & (
            stop_times_df['arrival_time'] < pd.to_timedelta('02:00:00')), 'arrival_time'] += pd.to_timedelta('1 days')
    return None


def finding_waiting(orig, dest, route, departure_time):
    global stop_times_df, stops_df, route_name_trips_dict, stops_trips_dict, stops_dict, route_name_trips_dict
    orig_id, orig_zone = stops_dict[orig]['stop_id'], stops_dict[orig]['zone_id']
    dest_id, dest_zone = stops_dict[dest]['stop_id'], stops_dict[dest]['zone_id']
    data = stop_times_df[
        stop_times_df.trip_id.isin(
            stops_trips_dict[orig_id] & stops_trips_dict[dest_id] & route_name_trips_dict[route])]
    data = data[data.stop_id.isin([orig_id, dest_id])]
    data.sort_values(by=['arrival_time'], inplace=True)
    data_org = data.groupby('trip_id').head(1)
    data_org = data_org[data_org.stop_id == orig_id]
    data_org = data_org[data_org.arrival_time >= departure_time]
    trips = list(data_org.trip_id.values)
    try:
        trip_id = trips[0]
        wait_time = (pd.to_timedelta(data_org[data_org.trip_id == trip_id].arrival_time.values[0]) - pd.to_timedelta(
            departure_time)).seconds / 60
        dept_time = pd.to_timedelta(data[data.trip_id == trip_id].arrival_time.values[1])
    except IndexError:
        wait_time = -10000
        dept_time = departure_time
    return wait_time, dept_time


def calculating_for_each_row(row):
    global od_data_2019, stops_dict, stop_times_df, trips_df, routes_df, issue_file
    wait_dict = {}
    for alt in range(1, 22):
        wait_dict[f'initial_wait_time_{alt}'] = 0
        wait_dict[f'transfer_time_{alt}'] = 0
        i = 0
        dep_time = row['departure_time']
        if row[f'route_{alt}'] != '0' and row[f'route_{alt}'] != 0 and row[f'route_{alt}'] != np.nan \
                and row[f'route_{alt}'] != None:
            legs = row[f'route_{alt}'].split('|')
            for leg in legs:
                orig, dest, route_code = re.split(r'>|@', leg)
                wait_time, dep1_time = finding_waiting(orig, dest, route_code, dep_time)
                dep_time = dep1_time
                i += 1
            if i == 1:
                wait_dict[f'initial_wait_time_{alt}'] = wait_time
            else:
                wait_dict[f'transfer_time_{alt}'] += wait_time
        wait_dict[f'waiting_time_{alt}'] = wait_dict[f'initial_wait_time_{alt}'] + wait_dict[f'transfer_time_{alt}']
    return wait_dict


if __name__ == '__main__':
    read_GTFS_data('./GTFS.zip')
    Reading_od_data('./Route_choice_Modelling/NBT19MTT3a_routechoice_tb_wf.csv')
    file_df = pd.read_csv('./Route_choice_Modelling/Output/test_2019_1.csv')
    file_df['departure_time'] = pd.to_timedelta(file_df['departure_time'])
    file_df.loc[(file_df['departure_time'] >= pd.to_timedelta('00:00:00')) & (
            file_df['departure_time'] < pd.to_timedelta('02:00:00')), 'departure_time'] += pd.to_timedelta('1 days')
    file_dict = file_df.loc[:].to_dict('records')
    # results = calculating_for_each_row(file_dict[0],21)
    CORES = 15
    start_time = time.time()
    with Pool(CORES) as pool:
        results = pool.map(calculating_for_each_row, file_dict)
    results_df = pd.DataFrame(results)
    # Print the results DataFrame
    print(results_df)
    file_df = pd.concat([file_df.loc[:, :], results_df], axis=1)
    file_df.to_csv('./Route_choice_Modelling/Output/test_2019_1_MNL.csv')
    print(f'Time taken for the parallel with 15 cores: {(time.time() - start_time)}')

# if __name__ == '__main__':
#     read_GTFS_data('./GTFS.zip')
#     Reading_od_data('./Route_choice_Modelling/NBT19MTT3a_routechoice_tb_wf.csv')
#     files = ['train_2019_2', 'train_2019_3', 'train_2019_4', 'train_2019_5', 'train_2019_6', 'test_2019_2',
#              'test_2019_3', 'test_2019_4', 'test_2019_5', 'test_2019_6']
#     for file in files:
#         print(file)
#         file_df = pd.read_csv(f'./Route_choice_Modelling/Output/{file}.csv')
#         file_df['departure_time'] = pd.to_timedelta(file_df['departure_time'])
#         file_df.loc[(file_df['departure_time'] >= pd.to_timedelta('00:00:00')) & (
#                 file_df['departure_time'] < pd.to_timedelta('02:00:00')), 'departure_time'] += pd.to_timedelta('1 days')
#         file_dict = file_df.loc[:].to_dict('records')
#         # results = calculating_for_each_row(file_dict[0],21)
#         CORES = 15
#         start_time = time.time()
#         with Pool(CORES) as pool:
#             results = pool.map(calculating_for_each_row, file_dict)
#         results_df = pd.DataFrame(results)
#         # Print the results DataFrame
#         file_df = pd.concat([file_df.loc[:, :], results_df], axis=1)
#         file_df.to_csv(f'./Route_choice_Modelling/Output/{file}_MNL.csv')
#         print('saved')
#         print(f'Time taken for the parallel with 15 cores: {(time.time() - start_time)}')
