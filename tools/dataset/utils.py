import json
import time
import argparse
import multiprocessing
from collections import OrderedDict
import lxml.etree as ET

import math
import numpy as np

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption

ALL_TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
TRIGGER_THRESHOLD = 2.0  # Threshold to say if a trigger position is new or repeated, works for matching positions
TRIGGER_ANGLE_THRESHOLD = 10  # Threshold to say if two angles can be considering matching when matching transforms
MAX_LEN = 120 # Max route length in meters
ID_START = 0 # Route ID to begin file


def gen_skeleton_dict(towns_, scenarios_):

    def scenarios_list():
        scen_type_dict_lst = []
        for scenario_ in scenarios_:
            scen_type_dict = {}
            scen_type_dict['available_event_configurations'] = []
            scen_type_dict['scenario_type'] = scenario_
            scen_type_dict_lst.append(scen_type_dict)
        return scen_type_dict_lst

    skeleton = \
        {
            "available_scenarios" : []
        }

    for town_ in towns_:
        skeleton["available_scenarios"].append({town_: scenarios_list()})

    return skeleton


def gen_scenarios(args, scenario_type_dict, town_scenario_tp_gen):
    if args.towns == 'all':
        towns = ALL_TOWNS 
    else:
        towns = [args.towns]
    
    for town_ in towns:
        client = carla.Client('localhost', 2000)
        client.set_timeout(200.0)
        world = client.load_world(town_)
        carla_map = world.get_map()  
        save_dir = args.save_dir
        for scen_type, _ in scenario_type_dict.items():
            town_scenario_tp_gen(town_, carla_map , scen_type, save_dir, world)


def sample_junctions(world_map, route, scenarios_list, town, start_dist=20, end_dist=20, min_len=50, max_len=MAX_LEN):
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
    for index in range(start_dist, len(route)-end_dist):
        if route[index-1][1] == RoadOption.LANEFOLLOW and route[index][1] != RoadOption.LANEFOLLOW:
            start_id = index-start_dist # start from before intersection to spawn scenarios
        elif start_id != -1 and route[index][1] == RoadOption.LANEFOLLOW:
            end_id = index + end_dist
            if end_id > start_id + min_len: # at least 50m distance
                
                # extra check to make sure that long/invalid routes are discarded
                start_wp = carla.Location(x=route[start_id][0].location.x, 
                            y=route[start_id][0].location.y, z=route[start_id][0].location.z)
                end_wp = carla.Location(x=route[end_id][0].location.x, 
                            y=route[end_id][0].location.y, z=route[end_id][0].location.z)
                waypoint_list = [start_wp, end_wp]
                extended_route = interpolate_trajectory(world_map, waypoint_list)
                potential_scenarios_definitions, _ = scan_route_for_scenarios(town, extended_route, scenarios_list)

                # skip long/invalid routes and routes with no scenario
                if len(extended_route) > max_len or len(extended_route) == 1 or len(potential_scenarios_definitions)==0:
                    start_id = -1
                    end_id = -1
                    continue

                downsampled_route = downsample_route(extended_route, 50) # mimic leaderboard downsampling

                custom_route = []
                for element in downsampled_route:
                    custom_transform = (route[start_id+element][0].location.x, route[start_id+element][0].location.y, 
                                        route[start_id+element][0].location.z, route[start_id+element][0].rotation.yaw)
                    custom_route.append(custom_transform)

                custom_routes.append(custom_route)
                
            start_id = -1
            end_id = -1

    return custom_routes

def process_route(world_map, route, scenarios_list, return_dict):
    interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
    wp_list = sample_junctions(world_map, interpolated_route, scenarios_list, route['town_name'])
    print ('got {} junctions in route {} (interpolated {} waypoints to {} waypoints)'.format(
                    len(wp_list), route['id'], len(route['trajectory']), len(interpolated_route)))

    return_dict[route['id']] = {'wp_list': wp_list, 'town_name': route['town_name'], 'length': len(interpolated_route)}


def main(args):

    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)

    routes_list = parse_routes_file(args.routes_file)
    scenarios_list = parse_annotations_file(args.scenarios_file)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    st = time.time()
    for index, route in enumerate(routes_list):
        if index == 0 or routes_list[index]['town_name'] != routes_list[index-1]['town_name']:
            world = client.load_world(route['town_name'])
            world_map = world.get_map()
        p = multiprocessing.Process(target=process_route, args=(world_map, route, scenarios_list, return_dict))
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
            for node in wps:
                ET.SubElement(add_route, 'waypoint', x='%f'%node[0], y='%f'%node[1],  z='%f'%node[2], 
                                                            pitch='0.0', roll='0.0', yaw='%f'%node[3])
            route_id += 1

    print('\nSource File:')
    print ('mean distance: ', np.array(route_lengths).mean())
    print ('median distance: ', np.median(np.array(route_lengths)))
        
    if args.save_file is not None:
        tree = ET.ElementTree(root)
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)

        new_index = 0
        outliers = 0
        route_lengths = []
        duplicate_list = []

        new_routes_list = parse_routes_file(args.save_file)
        if args.duplicate_file:
            duplicate_file_list = parse_routes_file(args.duplicate_file)
            for index, route in enumerate(duplicate_file_list):
                if index == 0 or duplicate_file_list[index]['town_name'] != duplicate_file_list[index-1]['town_name']:
                    world = client.load_world(route['town_name'])
                    world_map = world.get_map()
                new_interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
                locations = (new_interpolated_route[0][0].location.x, new_interpolated_route[0][0].location.y,
                            new_interpolated_route[-1][0].location.x, new_interpolated_route[-1][0].location.y)
                duplicate_list.append(locations)

        for index, route in enumerate(new_routes_list):
            if index == 0 or new_routes_list[index]['town_name'] != new_routes_list[index-1]['town_name']:
                world = client.load_world(route['town_name'])
                world_map = world.get_map()
            new_interpolated_route = interpolate_trajectory(world_map, route['trajectory'])
            current_node = root.getchildren()[index-outliers]
            locations = (new_interpolated_route[0][0].location.x, new_interpolated_route[0][0].location.y,
                        new_interpolated_route[-1][0].location.x, new_interpolated_route[-1][0].location.y)
            if (len(new_interpolated_route) > MAX_LEN) or (locations in duplicate_list):
                root.remove(current_node)
                outliers += 1
            else:
                duplicate_list.append(locations)
                route_lengths.append(len(new_interpolated_route))
                current_node.set("id", '%d'%(ID_START+new_index))
                new_index += 1

        tree = ET.ElementTree(root)
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)

        new_routes_list = parse_routes_file(args.save_file)

        print('\nTarget File:')
        print ('saved junctions: ', len(route_lengths))
        print ('outliers/duplicates: ', outliers)
        print ('file num junctions: ', len(new_routes_list))
        print ('mean distance: ', np.array(route_lengths).mean())
        print ('median distance: ', np.median(np.array(route_lengths)))


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


def parse_annotations_file(annotation_filename):
    """
    Return the annotations of which positions where the scenarios are going to happen.
    :param annotation_filename: the filename for the anotations file
    :return:
    """
    with open(annotation_filename, 'r') as f:
        annotation_dict = json.loads(f.read(), object_pairs_hook=OrderedDict)

    final_dict = OrderedDict()

    for town_dict in annotation_dict['available_scenarios']:
        final_dict.update(town_dict)

    return final_dict  # the file has a current maps name that is an one element vec


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


def downsample_route(route, sample_factor):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # Lane changing
        if curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When road option changes
        elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i-1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


def scan_route_for_scenarios(route_name, trajectory, world_annotations):
        """
        Just returns a plain list of possible scenarios that can happen in this route by matching
        the locations from the scenario into the route description

        :return:  A list of scenario definitions with their correspondent parameters
        """

        # the triggers dictionaries:
        existent_triggers = OrderedDict()
        # We have a table of IDs and trigger positions associated
        possible_scenarios = OrderedDict()

        # Keep track of the trigger ids being added
        latest_trigger_id = 0

        for town_name in world_annotations.keys():
            if town_name != route_name:
                continue

            scenarios = world_annotations[town_name]
            for scenario in scenarios:  # For each existent scenario
                scenario_name = scenario["scenario_type"]
                for event in scenario["available_event_configurations"]:
                    waypoint = event['transform']  # trigger point of this scenario
                    convert_waypoint_float(waypoint)
                    # We match trigger point to the  route, now we need to check if the route affects
                    match_position = match_world_location_to_route(
                        waypoint, trajectory)
                    if match_position is not None:
                        # We match a location for this scenario, create a scenario object so this scenario
                        # can be instantiated later

                        if 'other_actors' in event:
                            other_vehicles = event['other_actors']
                        else:
                            other_vehicles = None
                        scenario_subtype = get_scenario_type(scenario_name, match_position,
                                                                         trajectory)
                        if scenario_subtype is None:
                            continue
                        scenario_description = {
                            'name': scenario_name,
                            'other_actors': other_vehicles,
                            'trigger_position': waypoint,
                            'scenario_type': scenario_subtype, # some scenarios have route dependent configurations
                        }

                        trigger_id = check_trigger_position(waypoint, existent_triggers)
                        if trigger_id is None:
                            # This trigger does not exist create a new reference on existent triggers
                            existent_triggers.update({latest_trigger_id: waypoint})
                            # Update a reference for this trigger on the possible scenarios
                            possible_scenarios.update({latest_trigger_id: []})
                            trigger_id = latest_trigger_id
                            # Increment the latest trigger
                            latest_trigger_id += 1

                        possible_scenarios[trigger_id].append(scenario_description)

        return possible_scenarios, existent_triggers


def convert_waypoint_float(waypoint):
    """
    Convert waypoint values to float
    """
    waypoint['x'] = float(waypoint['x'])
    waypoint['y'] = float(waypoint['y'])
    waypoint['z'] = float(waypoint['z'])
    waypoint['yaw'] = float(waypoint['yaw'])


def match_world_location_to_route(world_location, route_description):
    """
    We match this location to a given route.
        world_location:
        route_description:
    """
    def match_waypoints(waypoint1, wtransform):
        """
        Check if waypoint1 and wtransform are similar
        """
        dx = float(waypoint1['x']) - wtransform.location.x
        dy = float(waypoint1['y']) - wtransform.location.y
        dz = float(waypoint1['z']) - wtransform.location.z
        dpos = math.sqrt(dx * dx + dy * dy + dz * dz)

        dyaw = (float(waypoint1['yaw']) - wtransform.rotation.yaw) % 360
        return dpos < TRIGGER_THRESHOLD \
            and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD))

    match_position = 0
    # TODO this function can be optimized to run on Log(N) time
    for route_waypoint in route_description:
        if match_waypoints(world_location, route_waypoint[0]):
            return match_position
        match_position += 1
    
    return None


def get_scenario_type(scenario, match_position, trajectory):
    """
    Some scenarios have different types depending on the route.
    :param scenario: the scenario name
    :param match_position: the matching position for the scenarion
    :param trajectory: the route trajectory the ego is following
    :return: tag representing this subtype

    Also used to check which are not viable (Such as an scenario
    that triggers when turning but the route doesnt')
    WARNING: These tags are used at:
        - VehicleTurningRoute
        - SignalJunctionCrossingRoute
    and changes to these tags will affect them
    """

    def check_this_waypoint(tuple_wp_turn):
        """
        Decides whether or not the waypoint will define the scenario behavior
        """
        if RoadOption.LANEFOLLOW == tuple_wp_turn[1]:
            return False
        elif RoadOption.CHANGELANELEFT == tuple_wp_turn[1]:
            return False
        elif RoadOption.CHANGELANERIGHT == tuple_wp_turn[1]:
            return False
        return True

    # Unused tag for the rest of scenarios,
    # can't be None as they are still valid scenarios
    subtype = 'valid'

    if scenario == 'Scenario4':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]:
                    subtype = 'S4left'
                elif RoadOption.RIGHT == tuple_wp_turn[1]:
                    subtype = 'S4right'
                else:
                    subtype = None
                break  # Avoid checking all of them
            subtype = None

    if scenario == 'Scenario7':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]:
                    subtype = 'S7left'
                elif RoadOption.RIGHT == tuple_wp_turn[1]:
                    subtype = 'S7right'
                elif RoadOption.STRAIGHT == tuple_wp_turn[1]:
                    subtype = 'S7opposite'
                else:
                    subtype = None
                break  # Avoid checking all of them
            subtype = None

    if scenario == 'Scenario8':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.LEFT == tuple_wp_turn[1]:
                    subtype = 'S8left'
                else:
                    subtype = None
                break  # Avoid checking all of them
            subtype = None

    if scenario == 'Scenario9':
        for tuple_wp_turn in trajectory[match_position:]:
            if check_this_waypoint(tuple_wp_turn):
                if RoadOption.RIGHT == tuple_wp_turn[1]:
                    subtype = 'S9right'
                else:
                    subtype = None
                break  # Avoid checking all of them
            subtype = None

    return subtype


def check_trigger_position(new_trigger, existing_triggers):
    """
    Check if this trigger position already exists or if it is a new one.
    :param new_trigger:
    :param existing_triggers:
    :return:
    """

    for trigger_id in existing_triggers.keys():
        trigger = existing_triggers[trigger_id]
        dx = trigger['x'] - new_trigger['x']
        dy = trigger['y'] - new_trigger['y']
        distance = math.sqrt(dx * dx + dy * dy)

        dyaw = (trigger['yaw'] - new_trigger['yaw']) % 360
        if distance < TRIGGER_THRESHOLD \
            and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD)):
            return trigger_id

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--routes_file', type=str, required=True, help='file containing the route waypoints')
    parser.add_argument('--scenarios_file', type=str, default='leaderboard/data/scenarios/eval_scenarios.json', help='file containing the scenarios')
    parser.add_argument('--save_file', type=str, required=False, default=None, help='xml file path to save the route waypoints')
    parser.add_argument('--duplicate_file', type=str, required=False, default=None, help='file to use for duplicate removal')
    args = parser.parse_args()

    main(args)