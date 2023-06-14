import time
import pandas as pd
from zipfile import ZipFile
from tqdm import tqdm
import warnings
from multiprocessing import Pool
import ast

warnings.filterwarnings("ignore")


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
    return None


def Reading_od_data(file_name='./Route_choice_Modelling/Route choice by origin-destination pair 2017.xls'):
    global routes_df, trips_df, stop_times_df, stops_df, stops_trips_dict, stops_dict, od_data_2017, route_name_trips_dict
    # Reading 2017 od data
    od_data_2017 = pd.read_excel(file_name, skiprows=3)
    od_data_2017.columns = ['origin_id', 'origin_name', 'destination_id', 'destination_name', 'transfer1', 'transfer2',
                            'transfer3', 'transfer4', '1', '2', '3', '4', '5', '6', 'total']
    # Removing the spaces at the end of the name.
    od_data_2017['origin_name'] = list(od_data_2017['origin_name'].apply(lambda x: x.rstrip()))
    od_data_2017['destination_name'] = list(od_data_2017['destination_name'].apply(lambda x: x.rstrip()))
    od_data_2017['transfer1'] = list(od_data_2017['transfer1'].apply(lambda x: x.rstrip()))
    od_data_2017['transfer2'] = list(od_data_2017['transfer2'].apply(lambda x: x.rstrip()))
    od_data_2017['transfer3'] = list(od_data_2017['transfer3'].apply(lambda x: x.rstrip()))
    od_data_2017['transfer4'] = list(od_data_2017['transfer4'].apply(lambda x: x.rstrip()))

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
    # Creating no of legs column
    od_data_2017['No_of_legs'] = 0
    # Creating the no of legs column based on transfers
    for j in range(1, 5):
        od_data_2017.loc[(od_data_2017[f'transfer{j}'] != '') & (od_data_2017.No_of_legs == 0), 'No_of_legs'] = j
    # Filtering the data
    stops_df.set_index('stop_name', inplace=True)
    stops = list(stops_df.index)
    print(len(od_data_2017))
    # Checking origin and destination are in the LU stops
    od_data_2017 = od_data_2017[od_data_2017.origin_name.isin(stops) & od_data_2017.destination_name.isin(stops)]
    print(len(od_data_2017))
    # Checking transfer stops are in the LU stops
    for j in range(1, 5):
        od_data_2017 = od_data_2017[(od_data_2017[f'transfer{j}'] == '') | (od_data_2017[f'transfer{j}'].isin(stops))]
    # Adding the route choice number
    od_data_2017['route_choice'] = od_data_2017.groupby(by=['origin_name', 'destination_name']).cumcount() + 1
    od_data_2017['merge_column'] = od_data_2017['origin_name'] + "-" + od_data_2017['destination_name'] + "-" + \
                                   od_data_2017['route_choice'].astype(str)
    stops_dict = stops_df[['stop_id', 'zone_id']].to_dict(orient='index')
    stop_times_df['arrival_time'] = pd.to_timedelta(stop_times_df['arrival_time'])
    trips_df['route_name'] = trips_df.merge(routes_df, left_on='route_id', right_on='route_id', how='left')[
        'route_short_name']
    stops_trips_dict = {}
    for stop in stops:
        stops_trips_dict[stop] = set(
            stop_times_df[stop_times_df.stop_id == stops_dict[stop]['stop_id']].trip_id.unique())
    stop_times_df.loc[(stop_times_df['arrival_time'] >= pd.to_timedelta('00:00:00')) & (
            stop_times_df['arrival_time'] < pd.to_timedelta('02:00:00')), 'arrival_time'] += pd.to_timedelta('1 days')
    route_name_trips_dict = {}
    for route in routes_df.route_short_name.unique():
        route_name_trips_dict[route] = set(trips_df[trips_df.route_name == route].trip_id.unique())
    return None


def finding_waiting(orig, dest, route, departure_time):
    global stop_times_df, stops_df, route_name_trips_dict,stops_dict
    orig_id, orig_zone = stops_dict[orig]['stop_id'], stops_dict[orig]['zone_id']
    dest_id, dest_zone = stops_dict[dest]['stop_id'], stops_dict[dest]['zone_id']
    data = stop_times_df[
        stop_times_df.trip_id.isin(stops_trips_dict[orig] & stops_trips_dict[dest] & route_name_trips_dict[route])]
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
        dept_time = dept_time
    return wait_time, dept_time


def calculating_for_each_row(row):
    wait_dict = {}
    for alt in range(1, 9):
        wait_dict[f'initial_wait_time_{alt}'] = 0
        wait_dict[f'transfer_time_{alt}'] = 0
        i = 0
        dep_time = row['departure_time']
        if row[f'route_{alt}'] != 0:
            for leg in row[f'route_{alt}']:
                orig = leg[0]
                dest = leg[1]
                route_code = leg[2]
                wait_time, dep1_time = finding_waiting(orig, dest, route_code, dep_time)
                i += 1
                if i == 1:
                    wait_dict[f'initial_wait_time_{alt}'] = wait_time
                else:
                    wait_dict[f'transfer_time_{alt}'] += wait_time
                    dep_time = dep1_time
        wait_dict[f'waiting_time_{alt}'] = wait_dict[f'initial_wait_time_{alt}'] + wait_dict[f'transfer_time_{alt}']
    return wait_dict


if __name__ == '__main__':
    read_GTFS_data('./GTFS.zip')
    Reading_od_data('./Route_choice_Modelling/Route choice by origin-destination pair 2017.xls')
    file_df = pd.read_csv('./Route_choice_Modelling/Output/test_2017_2.csv')
    file_df['departure_time'] = pd.to_timedelta(file_df['departure_time'])
    file_df.loc[(file_df['departure_time'] >= pd.to_timedelta('00:00:00')) & (
            file_df['departure_time'] < pd.to_timedelta('02:00:00')), 'departure_time'] += pd.to_timedelta('1 days')
    # Converting the strings to the list
    for alt in range(1, 9):
        file_df.loc[file_df[f'route_{alt}'] != 0, f'route_{alt}'] = file_df.loc[
            file_df[f'route_{alt}'] != 0, f'route_{alt}'].apply(lambda x: ast.literal_eval(x))
    # Converting dataframe to a list of dictionaries where each dictionary is a row.
    file_dict = file_df.loc[:].to_dict('records')
    CORES = 10
    start_time = time.time()
    with Pool(CORES) as pool:
        results = pool.map(calculating_for_each_row, file_dict)
    results_df = pd.DataFrame(results)
    # Print the results DataFrame
    file_df = pd.concat([file_df.loc[:,:], results_df], axis=1)
    file_df.to_csv('./Route_choice_Modelling/Output/test_2017_2_MNL.csv')
    print(f'Time taken for the parallel with 10 cores: {(time.time() - start_time)}')
