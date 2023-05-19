from collections import Counter
import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit
import os
import folium
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


def initialize_data(path: str) -> None:
    """
        This function initializes the data by reading the stops.csv file and filter the data.
    :param path: Directory path of the data : str
    :return: None
    """
    global merged_df
    # Read the stops file from Nihal's data
    bus_stops1_df = pd.read_csv(path + "/bmtc-stop-name-latlong-mapping.csv")
    bus_stops1_df.rename(columns={'formatted_stop_name': 'bus_stop_name', 'latitude': 'latitude_current',
                                  'longitude': 'longitude_current'}, inplace=True)
    bus_stops2_df = pd.read_csv(path + "/BMTC Routes Stops 2022 (1).csv")
    bus_stops3_df = pd.read_csv(path + '/bus_stop.csv')
    bus_stops4_df = pd.read_csv(path + '/retrived_bus_stops.csv')
    merged_df = pd.concat([bus_stops1_df[['bus_stop_name', 'latitude_current', 'longitude_current']],
                           bus_stops2_df[['bus_stop_name', 'latitude_current', 'longitude_current']]])
    merged_df = pd.concat([merged_df, bus_stops3_df[['bus_stop_name', 'latitude_current', 'longitude_current']]])
    merged_df = pd.concat([merged_df, bus_stops4_df[['bus_stop_name', 'latitude_current', 'longitude_current']]])
    # Latitude column_name is 'latitude_current' and longitude column is longitude_current
    print(f'No of bus stops initially :{len(merged_df)}')
    # Remove the duplicates of stop names and coordinates
    merged_df = merged_df.drop_duplicates(['bus_stop_name', 'latitude_current', 'longitude_current'])
    print('Removed duplicates of stops with same name and coordinates')
    print(f'No of bus stops after removing duplicates :{len(merged_df)}')
    # Remove the stops which are out of bounding box with respect to Bangalore
    merged_df = merged_df[((merged_df.latitude_current > 11) & (merged_df.latitude_current < 15)) & (
            (merged_df.longitude_current > 76) & (merged_df.longitude_current < 79))]
    print('Bounding box is applied to filter the bus stops')
    print(f'No of bus stops after applying Bounding box :{len(merged_df)}')
    return None


def finding_repeated_stop() -> tuple:
    """
        This function finds the repeated stops and returns the repeated stops, number of repeated stops and dictionary
    :return: tuple
    """
    global merged_df
    counter = Counter(merged_df.bus_stop_name)
    # Finding the repeated stops_names
    repeated_stops = []
    number = 0
    for stop in counter.keys():
        if counter[stop] > 1:
            repeated_stops.append(stop)
            number += counter[stop]
    return repeated_stops, number, counter


def statistics_repeated_stops(output_path, save: str = 'N') -> pd.DataFrame:
    """
        This function calculates the mean and standard deviation of the distance between the repeated stops
    :param output_path: Directory path to save the output csv file : str
    :param save: To save the output dataframe as csv file : str
    :return: pd.DataFrame
    """
    global merged_df
    repeated_stops, number, counter = finding_repeated_stop()
    output_dict = {'bus_stop_name': [], 'mean': [], 'std_dev': [], 'count': []}
    for stop in repeated_stops:
        stop_lat_long_array = np.array(merged_df
                                       [merged_df.bus_stop_name == stop][['latitude_current', 'longitude_current']])
        matrix = haversine_vector(stop_lat_long_array, stop_lat_long_array, Unit.METERS, comb=True)
        # Mask the diagonal elements
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        non_diag_elements = matrix[mask]
        # Calculate the mean and standard deviation of non-diagonal elements
        mean = np.round(np.mean(non_diag_elements), 2)
        std = np.round(np.std(non_diag_elements), 2)
        # Appending to the dictionary
        output_dict['bus_stop_name'].append(stop)
        output_dict['mean'].append(mean)
        output_dict['std_dev'].append(std)
        output_dict['count'].append(counter[stop])
    # Creating the dataframe and sorting it by mean
    output_df = pd.DataFrame(output_dict).sort_values('mean', ascending=False)
    if save.upper() == 'Y':
        output_df.to_csv(output_path, index=False)
    return output_df


def plot_coordinates(lat: list, long: list) -> folium.Map:
    """
        This function plots the coordinates on the map
    :param lat: list of latitude :list
    :param long:  of longitude :list
    :return: folium.Map
    """
    # Create a Folium map centered on the first coordinate
    map = folium.Map(location=(lat[0], long[0]), zoom_start=14)
    # Add markers for each coordinate
    for i in range(len(lat)):
        folium.Marker(location=(lat[i], long[i])).add_to(map)
    # Display the map
    return map


def creating_plots(path: str) -> None:
    """
        This function creates the plots of the stops with top 10 mean using plot_coordinates function
    :param path: str
    :return: None
    """
    global merged_df
    output_df = statistics_repeated_stops('', 'N')
    # Considering the stops with top 10 mean for plotting
    stops_plot = list(output_df.head(10)['bus_stop_name'])
    for stop in stops_plot:
        lats = list(merged_df[merged_df.bus_stop_name == stop].latitude_current)
        lons = list(merged_df[merged_df.bus_stop_name == stop].longitude_current)
        map = plot_coordinates(lats, lons)
        map.save(path + '/trip-planner-logs/plots' + f'/{stop}.html')
    return None


def random_selection() -> pd.DataFrame:
    """
        This function randomly selects one row for each group of bus_stop_name and stored in dataframe
    :return: pd.DataFrame
    """
    global merged_df, unique_bus_stops
    # Use groupby() and sample to randomly select one row for each group
    unique_bus_stops = merged_df.groupby('bus_stop_name').sample(n=1)
    # Reset the index of the result DataFrame
    unique_bus_stops = unique_bus_stops.reset_index(drop=True)
    return unique_bus_stops


def update_trip_planner(input_path: str, output_path: str) -> tuple:
    """
        This function adds new coordinates of origin and destination
        stops to the trip planner data and returns no of missing row coordinates
    :param input_path: Path of the trip planner csv file : str
    :param output_path: Path of the updated csv file is saved : str
    :return: no of missing row coordinates  and name of stops: tuple
    """
    unique_bus_stops = random_selection()
    # Creating the trip planner dataframe
    trip_planner_df = pd.read_csv(input_path)
    # Merging the trip_planner_df with unique_bus_stops only lat and long for origin stops
    trip_planner_df = trip_planner_df.merge(
        unique_bus_stops[['bus_stop_name', 'latitude_current', 'longitude_current']],
        left_on='orig_stop_name', right_on='bus_stop_name', how='left')
    trip_planner_df.rename(columns={'latitude_current': 'Origin_lat', 'longitude_current': 'Origin_long'}, inplace=True)
    trip_planner_df.drop(labels=['bus_stop_name'], axis=1, inplace=True)

    # Merging the trip_planner_df with unique_bus_stops with only lat and long for destination stops
    trip_planner_df = trip_planner_df.merge(
        unique_bus_stops[['bus_stop_name', 'latitude_current', 'longitude_current']],
        left_on='dest_stop_name', right_on='bus_stop_name', how='left')
    trip_planner_df.rename(columns={'latitude_current': 'desti_lat', 'longitude_current': 'desti_long'}, inplace=True)
    trip_planner_df.drop(labels=['bus_stop_name'], axis=1, inplace=True)
    # Calculating the missing desti_stops and orig_stops
    no_missing_od = len(trip_planner_df[trip_planner_df.desti_lat.isna() | trip_planner_df.Origin_lat.isna()])
    stops_coordinates_missing = list(trip_planner_df[trip_planner_df.desti_lat.isna()].dest_stop_name.unique())
    stops_coordinates_missing.extend(list(trip_planner_df[trip_planner_df.Origin_lat.isna()].orig_stop_name.unique()))
    # Saving the dataframe as csv
    try:
        trip_planner_df.to_csv(output_path, index=False)
    except OSError:
        # Specify the path where you want to create the directory
        directory_path = '/'.join(output_path.split('/')[:-1])
        os.makedirs(directory_path)
        trip_planner_df.to_csv(output_path, index=False)
    return no_missing_od, stops_coordinates_missing


def cleaning_data(path: str) -> None:
    """
    This function takes the data directory cleans the data and saves the cleaned data in its sub folder
    :param path: str
    :return: None
    """
    global merged_df, num_missing_stops, stops_missing_ls
    initialize_data(path)
    # Saving Mean and std of repeated stops in csv file in plot file located in trip-planner-logs
    statistics_repeated_stops(path + '/trip-planner-logs/plots/out.csv', save="Y")

    # Creates HTML files for top 10 stops with high mean
    creating_plots(path)
    print('HTML files are created')

    # Updating all trip-planner-logs csv files and plotting the no of rows missing in each log
    csv_files = glob.glob(os.path.join(path + '/trip-planner-logs', "*.csv"))
    num_missing_stops = []
    stops_missing_ls = []
    for file in tqdm(csv_files):
        file_ls = file.split('\\')
        output_file = path + '/' + 'Updated-trip-planner-logs' + '/' + file_ls[-1]
        no_missing_od, stops_coordinates_missing = update_trip_planner(file, output_file)
        num_missing_stops.append(no_missing_od)
        stops_missing_ls.extend(stops_coordinates_missing)
    stops_missing_ls = list(set(stops_missing_ls))

    # Plot size
    plt.figure(figsize=(19.27, 9.67))

    # Plot the curve
    plt.plot(list(range(len(num_missing_stops))), num_missing_stops)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('No of rows')
    plt.title('No of missing OD coordinate rows')
    plt.savefig(path + '/trip-planner-logs/plots/missing_rows.jpg')
    # Display the plot
    plt.show()
    print('Plot for no of missing data rows is created.')
    return None


if __name__ == '__main__':
    # In below function pass the path of the cleaning_bangalore_OD folder
    cleaning_data('./cleaning_bangalore_OD')
    # print(f"No of missing bus stop coordinates:{len(stops_missing_ls)}
#
# output_pd = pd.read_csv(r'./cleaning_bangalore_OD/trip-planner-logs/plots/out.csv')
# out
# # Bivariate analysis of count and mean of repeated stops
# sns.jointplot(x='count', y='mean', data=output_pd, kind='reg')
