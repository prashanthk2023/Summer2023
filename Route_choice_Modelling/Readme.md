# Route Choice Modelling Attribute Derivation For London Underground

## Description:
<p align = "justify"> For building an MNL model for route choice modelling, we required attributes for each available route. Route choices can be varied from person to person and their choices depend on the parameters like fare, total travel time, number of transfers, source waiting time and time between the transfers along the route. Since London Underground is a metro where the fare will not change with respect to the route for OD. So, no need to derive the fare for the metro. For the MNL model, the following attributes are considered and derived using the GTFS data set. </p>

* Number of Transfers
* Total Travel time
* Transfers time
* In-vehicle travel time
* Path Size

## Data:
GTFS data is scrapped from the [tfl.gov.uk](https://tfl.gov.uk/info-for/open-data-users/). The GTFS (General Transit Feed Specification) contains comprehensive data about various aspects of public transportation systems, including details about stops, trips, routes, stop times, fares, and more. Route choices are determined from the perspective of the agents, and since we lack individual commuter information, we utilized OD (Origin-Destination) data from the years 2019 and 2017. The OD data includes information about the available routes and the corresponding number of commuters opting for each route for each time interval. We utilized the OD data and transformed it into a commuter-level dataset, which includes information about chosen routes and available routes for each commuter. The departure time for each commuter was randomly generated within a specific time interval.

&nbsp; &nbsp; &nbsp; The route is represented as the $O > transfer_1 > transfer_2 > ... > D$, where O and D represent the origin and destination stops respectively and $transfer_i$ represents the $i^{th}$ transfer stop.


