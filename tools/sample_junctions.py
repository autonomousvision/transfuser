import os
import sys
import json
import time
import argparse
import multiprocessing
import lxml.etree as ET

import numpy as np
import matplotlib.pyplot as plt

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption

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

            # Waypoints is basically a list of XML nodes

        list_route_descriptions.append({
            'id': route_id,
            'town_name': route_town,
            'trajectory': waypoint_list
        })

    return list_route_descriptions


def sample_junctions(world_map, route):
    """
    Sample individual junctions from the interpolated routes
    Args:
        world_map: town map
        route: interpolated route
    Return:
        custom_routes: list of (start wp, end wp) each representing an individual junction
    """
    custom_routes = []
    start_id = -1
    end_id = -1
    for index in range(18, len(route)):
        if route[index-1][1] == RoadOption.LANEFOLLOW and route[index][1] != RoadOption.LANEFOLLOW:
            start_id = index-18 # start from 18m before rather than exactly at the intersection otherwise traffic lights don't get detected properly
        elif start_id != -1 and route[index][1] == RoadOption.LANEFOLLOW:
            end_id = index-1
            if end_id > start_id + 28: # at least 28m distance
                
                # extra check to make sure that long/invalid routes are discarded
                start_wp = carla.Location(x=route[start_id][0].location.x, 
                            y=route[start_id][0].location.y, z=route[start_id][0].location.z)
                end_wp = carla.Location(x=route[end_id][0].location.x, 
                            y=route[end_id][0].location.y, z=route[end_id][0].location.z)
                waypoint_list = [start_wp, end_wp]
                extended_route = interpolate_trajectory(world_map, waypoint_list)
                if len(extended_route) >= 100 or len(extended_route) == 1:
                    start_id = -1
                    end_id = -1
                    continue
                
                start_transform = (route[start_id][0].location.x, route[start_id][0].location.y, 
                                route[start_id][0].location.z, route[start_id][0].rotation.yaw)
                end_transform = (route[end_id][0].location.x, route[end_id][0].location.y, 
                                route[end_id][0].location.z, route[end_id][0].rotation.yaw)
                custom_routes.append([start_transform, end_transform])
                
            start_id = -1
            end_id = -1

    return custom_routes

def process_route(world_map, route, return_dict):
    interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
    wp_list = sample_junctions(world_map, interpolated_route)
    print ('got {} junctions in interpolated route {} from {} waypoints to {} waypoints'.format(
                    len(wp_list), route['id'], len(route['trajectory']), len(interpolated_route)))

    return_dict[route['id']] = {'wp_list': wp_list, 'town_name': route['town_name'], 'length': len(interpolated_route)}


def main(args):

    client = carla.Client('localhost', 2100)
    client.set_timeout(200.0)

    routes_list = parse_routes_file(args.routes_file)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    st = time.time()
    for index, route in enumerate(routes_list):
        if index == 0 or routes_list[index]['town_name'] != routes_list[index-1]['town_name']:
            world = client.load_world(route['town_name'])
            world_map = world.get_map()
        p = multiprocessing.Process(target=process_route, args=(world_map, route, return_dict))
        jobs.append(p)
        p.start()

    for process in jobs:
        process.join()
    print ('{} routes processed in {} seconds'.format(len(return_dict), time.time()-st))

    route_id = 0
    total_junctions = 0
    route_lengths = []
    root = ET.Element('routes')
    for curr_route in return_dict.keys():
        wp_list = return_dict[curr_route]['wp_list']
        town_name = return_dict[curr_route]['town_name']
        total_junctions += len(wp_list)
        route_lengths.append(return_dict[curr_route]['length'])
        for wps in wp_list:
            add_route = ET.SubElement(root, 'route', id='%d'%route_id, town=town_name) # 'town' in carla 0.9.10, 'map' in carla 0.9.9
            ET.SubElement(add_route, 'waypoint', x='%f'%wps[0][0], y='%f'%wps[0][1],  z='%f'%wps[0][2], 
                                                            pitch='0.0', roll='0.0', yaw='%f'%wps[0][3])
            ET.SubElement(add_route, 'waypoint', x='%f'%wps[1][0], y='%f'%wps[1][1], z='%f'%wps[1][2], 
                                                            pitch='0.0', roll='0.0', yaw='%f'%wps[1][3])
            route_id += 1

    print ('total_junctions: ', total_junctions)
    print ('mean distance: ', np.array(route_lengths).mean())
    print ('median distance: ', np.median(np.array(route_lengths)))
        
    if args.save_file is not None:
        tree = ET.ElementTree(root)
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--routes_file', type=str, required=True, help='file containing the route waypoints')
    parser.add_argument('--save_file', type=str, required=False, default=None, help='xml file path to save the route waypoints')

    args = parser.parse_args()

    main(args)