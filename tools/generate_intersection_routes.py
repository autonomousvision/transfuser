import glob
import os
import sys
import lxml.etree as ET
import argparse
import random
import time

import carla

SAMPLING_DISTANCE = [100]

def add_intersection(transform, root, route_id):
    '''
    Sample (start wp, end wp) pair along the canonical axes in a 100x100 grid
    Args:
        transform: carla transform of the grid center (position of traffic light)
        root: root of the xml tree structure
        route_id: route counter
    '''
    x, y, yaw = transform.location.x, transform.location.y, transform.rotation.yaw
    req_yaw = yaw + 180.0 # the vehicle should be opposite the traffic light
    for dist in SAMPLING_DISTANCE:
        for mul in [-1, 1]:
            route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
            ET.SubElement(route, 'waypoint', x='%f'%(x+mul*dist), y='%f'%(y), z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y+mul*dist), z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            route_id += 1

            route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
            ET.SubElement(route, 'waypoint', x='%f'%(x+mul*dist), y='%f'%y, z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y-mul*dist), z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            route_id += 1

            route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
            ET.SubElement(route, 'waypoint', x='%f'%(x-mul*dist), y='%f'%(y), z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y+mul*dist), z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            route_id += 1

            route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
            ET.SubElement(route, 'waypoint', x='%f'%(x+mul*dist), y='%f'%y, z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y-mul*dist), z='0.0', 
                                             pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
            route_id += 1

    return root, route_id

def add_intersection_subsample(transform, root, route_id):
    '''
    Same function as above but samples 75% fewer routes
    Args:
        transform: carla transform of the grid center (position of traffic light)
        root: root of the xml tree structure
        route_id: route counter
    '''
    x, y, yaw = transform.location.x, transform.location.y, transform.rotation.yaw
    req_yaw = yaw + 180.0 # the vehicle should be opposite the traffic light
    for dist in SAMPLING_DISTANCE:
        for mul in [-1, 1]:
            if random.randint(0,7) == 0:
                route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
                ET.SubElement(route, 'waypoint', x='%f'%(x+mul*dist), y='%f'%(y), z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y+mul*dist), z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                route_id += 1

            if random.randint(0,7) == 0:
                route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
                ET.SubElement(route, 'waypoint', x='%f'%(x+mul*dist), y='%f'%y, z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y-mul*dist), z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                route_id += 1

            if random.randint(0,7) == 0:
                route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
                ET.SubElement(route, 'waypoint', x='%f'%(x-mul*dist), y='%f'%(y), z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y+mul*dist), z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                route_id += 1

            if random.randint(0,7) == 0:
                route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
                ET.SubElement(route, 'waypoint', x='%f'%(x+mul*dist), y='%f'%y, z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                ET.SubElement(route, 'waypoint', x='%f'%(x), y='%f'%(y-mul*dist), z='0.0', 
                                                 pitch='0.0', roll='0.0', yaw='%f'%req_yaw)
                route_id += 1

    return root, route_id

def main():
    client = carla.Client('localhost', 2100)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    print ('loaded world')

    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print ('got %d traffic lights'%len(traffic_lights_list))

    # each traffic light group at an intersection counted once
    count = 0
    route_id = 0
    root = ET.Element('routes')
    traffic_light_visited = []
    for traffic_light in traffic_lights_list:
        if traffic_light.id not in traffic_light_visited:
            traffic_light_visited.append(traffic_light.id)
            count += 1
            if not args.subsample:
                root, route_id = add_intersection(traffic_light.get_transform(), root, route_id)
            else:
                root, route_id = add_intersection_subsample(traffic_light.get_transform(), root, route_id)
            for adjacent_traffic_light in traffic_light.get_group_traffic_lights():
                traffic_light_visited.append(adjacent_traffic_light.id)
    print ('unique intersections: ', count)

    tree = ET.ElementTree(root)

    if args.save_file is not None:
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)
    
    len_tree = 0
    for _ in tree.iter('route'):
        len_tree += 1
    print ('total routes: ', len_tree)


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_file', type=str, required=False, default=None, help='xml file path to save the route waypoints')
    parser.add_argument('--town', type=str, default='Town05', help='town for generating routes')
    parser.add_argument('--subsample', action='store_true', default=False, help='sample 75% fewer routes')
    
    args = parser.parse_args()

    main()
