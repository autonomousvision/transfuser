import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# navigational commands: RoadOption:LEFT and so on
NV_DICT = {1: 'LEFT', 2: 'RIGHT', 3: 'STRAIGHT', 4: 'LANEFOLLOW', 5: 'CHANGELANELEFT', 6: 'CHANGELANERIGHT'}

def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    Args:
        world: an reference to the CARLA world so we can use the planner
        waypoints_trajectory: the current coarse trajectory
        hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    Return: 
        route: full interpolated route both in GPS coordinates and also in its original form.
    """

    dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    # Obtain route plan
    route = []
    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.

        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))
            # print (wp_tuple[0].transform.location, wp_tuple[1])

    return route


def parse_routes_file(route_filename, single_route=None):
    """
    Returns a list of route elements that is where the challenge is going to happen.
    Args:
        route_filename: the path to a set of routes.
        single_route: If set, only this route shall be returned
    Return:
        list_route_descriptions: List of dicts containing the waypoints, id and town of the routes
    """

    list_route_descriptions = []
    tree = ET.parse(route_filename)
    for route in tree.iter("route"):
        route_town = route.attrib['town']
        route_id = route.attrib['id']
        if single_route and route_id != single_route:
            continue

        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
                                                y=float(waypoint.attrib['y']),
                                                z=float(waypoint.attrib['z'])))

        list_route_descriptions.append({
            'id': route_id,
            'town_name': route_town,
            'trajectory': waypoint_list
        })

    return list_route_descriptions


def main(args):

	client = carla.Client('localhost', 2100)
	client.set_timeout(200.0)

	routes_list = parse_routes_file(args.routes_file)

	anomaly = []
	total_waypoints = 0
	waypoint_threshold = args.wp_threshold
	nv_cnt = [0]*7 # total 6 navigational commands [1,6]

	for index, route in enumerate(routes_list):
		if index == 0 or routes_list[index]['town_name'] != routes_list[index-1]['town_name']:
			world = client.load_world(route['town_name'])
			world_map = world.get_map()
		interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
		# print ('interpolated route {} from {} waypoints to {} waypoints'.format(route['id'], 
		# 									len(route['trajectory']), len(interpolated_route)))
		total_waypoints += len(interpolated_route)
		if len(interpolated_route) >= waypoint_threshold:
			anomaly.append(route['id'])
			continue
		for waypoint in interpolated_route:
			nv_cnt[waypoint[1].value] += 1

	print ('found anomalies in routes ids: ', anomaly)

	print ('retained {:.3f} % data after discarding routes with more than {} waypoints'.format(100*np.sum(nv_cnt)/total_waypoints, waypoint_threshold))
	for index in range(1,7):
		print (NV_DICT[index], nv_cnt[index], nv_cnt[index]/np.sum(nv_cnt))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--routes_file', type=str, required=True, help='file containing the route waypoints')
	parser.add_argument('--wp_threshold', type=int, required=False, default=1e10)

	args = parser.parse_args()

	main(args)