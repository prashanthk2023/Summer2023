# Route Choice Modelling Attribute Derivation For London Underground

## Description:
<p align = "justify"> We required attributes for each available route to build an MNL model for route choice modelling. Route choices can be varied from person to person, and their choices depend on the parameters like fare, total travel time, number of transfers, source waiting time and time between the transfers along the route. Since London Underground is a metro, the fare will remain the same for the route for OD. So, no need to derive the fare for the metro. For the MNL model, the following attributes are considered and derived using the GTFS data set. </p>

* Number of Transfers
* Total Travel time
* Transfers time
* In-vehicle travel time
* Path Size

## Data:
GTFS data is scrapped from the [tfl.gov.uk](https://tfl.gov.uk/info-for/open-data-users/). The GTFS (General Transit Feed Specification) contains comprehensive data about various public transportation systems, including details about stops, trips, routes, stop times, fares, and more. Route choices are determined from the agent's perspective, and since we lack individual commuter information, we utilized OD (Origin-Destination) data from 2019 and 2017. The OD data includes information about the available routes and the corresponding number of commuters opting for each route for each time interval. We transformed the OD data into a commuter-level dataset, which includes information about chosen routes and available routes for each commuter. The departure time for each commuter was randomly generated within a specific time interval.

&nbsp; &nbsp; &nbsp; The route is represented as the $O > transfer_1 > transfer_2 > ... > D$, where O and D represent the origin and destination stops respectively, and $transfer_i$ represents the $i^{th}$ transfer stop.

## Methodology:
The total travel time is derived by computing the difference between the arrival time at the destination and the departure time from the origin. The in-vehicle travel time is determined by adding the travel time for each journey segment. Transfer time, on the other hand, refers to the waiting time between transfers and is calculated as the time between each segment. [Path size] {https://journals.sagepub.com/doi/10.3141/2538-02} is introduced to capture any correlation among the available route alternatives.




