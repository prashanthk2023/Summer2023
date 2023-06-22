import pandas as pd
from zipfile import ZipFile
import random


def read_GTFS(path):
    global stops_df, stop_times_df, trips_df, routes_df, calendar_df, shapes_df
    with ZipFile(path) as zip:
        with zip.open('BLR_14/routes.txt') as routes:
            routes_df = pd.read_csv(routes)
        with zip.open('BLR_14/trips.txt') as trips:
            trips_df = pd.read_csv(trips)
        with zip.open('BLR_14/stoptimes.txt') as stop_times:
            stop_times_df = pd.read_csv(stop_times)
        with zip.open('BLR_14/stops.txt') as stops:
            stops_df = pd.read_csv(stops)
        with zip.open('BLR_14/shapes.txt') as shapes:
            shapes_df = pd.read_csv(shapes)
    return None


def create_GTFS(path, no_of_trips, output_path):
    read_GTFS(path)
    actual_trips = list(trips_df.trip_id.unique())
    random_trips = random.sample(list(actual_trips), no_of_trips)
    # creating new trips df for the random trips
    new_trips_df = trips_df[trips_df.trip_id.isin(random_trips)]
    # Getting routes corresponding to the random trips
    new_routes = list(new_trips_df.route_id.unique())
    # creating new stop_times df for the random trips
    new_stop_times_df = stop_times_df[stop_times_df.trip_id.isin(random_trips)]
    # Getting stops present in the random trips
    new_stops = list(new_stop_times_df.stop_id.unique())
    # creating new routes df for the random trips
    new_routes_df = routes_df[routes_df.route_id.isin(new_routes)]
    # creating new stops df for the random trips
    new_stops_df = stops_df[stops_df.stop_id.isin(new_stops)]
    # Creating the GTFS dataset as zipfile
    with ZipFile(output_path, 'w') as zip:
        zip.writestr('routes.txt', new_routes_df.to_csv(index=False))
        zip.writestr('trips.txt', new_trips_df.to_csv(index=False))
        zip.writestr('stoptimes.txt', new_stop_times_df.to_csv(index=False))
        zip.writestr('stops.txt', new_stops_df.to_csv(index=False))
        zip.writestr('shapes.txt', shapes_df.to_csv(index=False))
    return None


def generating_queries(trip_plan_logger, number):
    return None


if __name__ == '__main__':
    create_GTFS('./MDVSP/BLR_14.zip', 10000, f'./MDVSP/BMTC_GTFS_{10000}.zip')
