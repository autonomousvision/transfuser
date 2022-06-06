# CARLA Leaderboard 0.9.10.1

This code is taken from the official leaderboard repository, with some minor changes for the Longest6 evaluation. The changes are made in a local copy of the original files, named `*_local.py`. The main changes include:

1. Dense traffic: https://github.com/autonomousvision/transfuser/blob/ff02e6fffa961e473a6a6486f387582349dc7f44/leaderboard/leaderboard/scenarios/route_scenario_local.py#L449
2. No penalty for stop infractions: https://github.com/autonomousvision/transfuser/blob/ff02e6fffa961e473a6a6486f387582349dc7f44/leaderboard/leaderboard/utils/statistics_manager_local.py#L26

## Original README.md
The main goal of the CARLA Autonomous Driving Leaderboard is to evaluate the driving proficiency of autonomous agents in realistic traffic situations. The leaderboard serves as an open platform for the community to perform fair and reproducible evaluations, simplifying the comparison between different approaches.

Autonomous agents have to drive through a set of predefined routes. For each route, agents are initialized at a starting point and have to drive to a destination point. The agents will be provided with a description of the route. Routes will happen in a variety of areas, including freeways, urban scenes, and residential districts.

Agents will face multiple traffic situations based in the NHTSA typology, such as:

* Lane merging
* Lane changing
* Negotiations at traffic intersections
* Negotiations at roundabouts
* Handling traffic lights and traffic signs
* Coping with pedestrians, cyclists and other elements

The user can change the weather of the simulation, allowing the evaluation of the agent in a variety of weather conditions, including daylight scenes, sunset, rain, fog, and night, among others.

More information can be found [here](https://leaderboard.carla.org/)
