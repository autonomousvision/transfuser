
import os
import argparse
import lxml.etree as ET

import math
import carla

from utils import ALL_TOWNS, \
        parse_annotations_file, interpolate_trajectory, scan_route_for_scenarios

ID_START = 0
MAX_LEN = 380


def main(args):
    route_id = ID_START
    road_type = args.road_type
    root = ET.Element('routes')

    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()

    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    scenarios_list = parse_annotations_file(args.scenarios_file)

    count_all_routes  = 0
    duplicates = 0
    if road_type == 'curved':
        DOT_PROD_SLACK = 0.02
        PRECISION = 2
        DISTANCE = 380 
        PRUNE_ROUTES_MIN_LEN = 20
        NUM_WAYPOINTS_DISTANCE = int(DISTANCE / PRECISION)
        MIN_ROUTE_LENGTH = 4
        duplicate_list = []
        

        for wp_it, waypoint in enumerate(topology):

            cur_wp = waypoint

            wp_list_nxt = [cur_wp]
            if not cur_wp.is_junction:    
                
                #forward wp
                while(True):
                    cur_wp_ = wp_list_nxt[-1]
                    try:
                        nxt_wp = cur_wp_.next(PRECISION)[0]
                    except:
                        break
                    
                    if not nxt_wp.is_junction:
                        wp_list_nxt.append(nxt_wp)
                    else:
                        break


            #backward_wp   
            wp_list_prev = [cur_wp]  
            if not cur_wp.is_junction:  
                while(True):
                    cur_wp_ = wp_list_prev[-1]
                    try:
                        nxt_wp = cur_wp_.previous(PRECISION)[0]
                    except:
                        break
                    if not nxt_wp.is_junction:
                        wp_list_prev.append(nxt_wp)
                    else:

                        break
            

            if len(wp_list_prev) + len(wp_list_nxt) > MIN_ROUTE_LENGTH:
                final_wps_list = list(reversed(wp_list_prev[1:])) +  wp_list_nxt
                cur_wp = final_wps_list[int(len(final_wps_list)/2)]

                prev_wp = final_wps_list[0]
                nxt_wp = final_wps_list[-1]
                vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                vec_wp_prev = cur_wp.transform.location -  prev_wp.transform.location

                norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y) #+ 
                
                try:
                    dot_ = ( vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
                except:
                    dot_ = -1
                
                if (dot_ > -1 - DOT_PROD_SLACK and dot_ < -1 +  DOT_PROD_SLACK):

                    continue
                else:
                    
                    truncated_wp_lst = []
                    
                    for i_ in range(len(final_wps_list)):
                        
                        tmp_wps =  final_wps_list[i_*NUM_WAYPOINTS_DISTANCE : i_*NUM_WAYPOINTS_DISTANCE + NUM_WAYPOINTS_DISTANCE]
                        if len(tmp_wps) > 1:
                            cur_wp = tmp_wps[int(len(tmp_wps)/2)]
                            prev_wp = tmp_wps[0]
                            nxt_wp = tmp_wps[-1]

                            vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                            vec_wp_prev = cur_wp.transform.location -  prev_wp.transform.location

                            norm_ = math.sqrt(vec_wp_nxt.x * vec_wp_nxt.x + vec_wp_nxt.y * vec_wp_nxt.y) * math.sqrt(vec_wp_prev.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_prev.y)
                            
                            try:
                                dot_ = ( vec_wp_nxt.x * vec_wp_prev.x + vec_wp_prev.y * vec_wp_nxt.y) / norm_
                            except:
                                dot_ = -1
                            
                            if not dot_ < -1 +  DOT_PROD_SLACK:
                                truncated_wp_lst.append(tmp_wps) 

                        locations =[]
                        for wps_sub in truncated_wp_lst:
                            locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))

                        are_loc_dups = []
                        for location_ in locations:
                            flag_cum_ctr = []
                            for loc_dp in duplicate_list:
                                flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc+ PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp) ] # threshold hardset
                                flag_AND_ctr = all(flag_ctrs)
                                flag_cum_ctr.append(flag_AND_ctr)
                            is_loc_dup = any(flag_cum_ctr)

                            are_loc_dups.append(is_loc_dup)
                        
                        for j_, wps_ in enumerate(truncated_wp_lst):
                            if not are_loc_dups[j_]:
                                count_all_routes +=1
                                duplicate_list.append(locations[j_])
                                wps_tmp =  [wps_[0].transform.location, wps_[-1].transform.location]
                                extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                potential_scenarios_definitions, _ = scan_route_for_scenarios(args.town, extended_route, scenarios_list)
                                
 
                                wps_ = [wps_[0], wps_[-1]]  

                                if (len(extended_route) <= MAX_LEN and len(potential_scenarios_definitions) > 0) and len(extended_route) > PRUNE_ROUTES_MIN_LEN:
                                    route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)

                                    for k_, wp_sub in enumerate(wps_):
                                        ET.SubElement(route, 'waypoint', x='%f'%(wp_sub.transform.location.x), y='%f'%(wp_sub.transform.location.y), z='0.0', 
                                                                pitch='0.0', roll='0.0', yaw='%f'%wp_sub.transform.rotation.yaw)

                                    route_id += 1
                            else:
                                duplicates +=1
        
    else:
        
        PRECISION = 2
        SAMPLING_DISTANCE = 30 
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = junc_.bounding_box
                
                j_wps = junc_.get_waypoints(carla.LaneType.Driving)
                
                for it_, j_wp in enumerate(j_wps):
                    wp_p = j_wp[0]
                    dist_prev = 0
                    wp_list_prev = []

                    while(True):
                        
                        wp_list_prev.append(wp_p)
                        
                        try:
                            wp_p = wp_p.previous(PRECISION)[0] # THIS ONLY CONSIDERS ONE ROUTE
                        except:
                            break
                        
                        dist_prev += PRECISION

                        if dist_prev> SAMPLING_DISTANCE:
                            break
                    

                    dist_nxt = 0
                    wp_n = j_wp[1]
                    wp_list_nxt = []

                    while(True):
                        wp_list_nxt.append(wp_n)
                        
                        try:
                            wp_n = wp_n.next(PRECISION)[0] # THIS ONLY CONSIDERS ONE ROUTE
                        except:
                            break
                        
                        dist_nxt += PRECISION
                        if dist_nxt > SAMPLING_DISTANCE:
                            break

                    final_wps_list = list(reversed(wp_list_prev[1:])) +  wp_list_nxt


                    truncated_wp_lst = [final_wps_list]
                    locations =[]
                    for wps_sub in truncated_wp_lst:
                        locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y))

                    are_loc_dups = []
                    for location_ in locations:
                        flag_cum_ctr = []
                        for loc_dp in duplicate_list:
                            flag_ctrs = [True if prev_loc - PRECISION <= curr_loc <= prev_loc+ PRECISION else False for curr_loc, prev_loc in zip(location_, loc_dp) ] # threshold hardset
                            flag_AND_ctr = all(flag_ctrs)
                            flag_cum_ctr.append(flag_AND_ctr)
                        is_loc_dup = any(flag_cum_ctr)

                        are_loc_dups.append(is_loc_dup)


                    for j_, wps_ in enumerate(truncated_wp_lst):
                        if not are_loc_dups[j_]:
                            count_all_routes +=1
                            duplicate_list.append(locations[j_])
                            wps_tmp =  [wps_[0].transform.location, wps_[-1].transform.location]
                            extended_route = interpolate_trajectory(carla_map, wps_tmp)
                            potential_scenarios_definitions, _ = scan_route_for_scenarios(args.town, extended_route, scenarios_list)
                            
                            if not len(potential_scenarios_definitions):
                                continue
    
                            wps_ = [wps_[0], wps_[-1]]                                

                            if  (len(extended_route) < MAX_LEN and len(potential_scenarios_definitions) > 0):
                                route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
                                for k_, wp_sub in enumerate(wps_):
                                    ET.SubElement(route, 'waypoint', x='%f'%(wp_sub.transform.location.x), y='%f'%(wp_sub.transform.location.y), z='0.0', 
                                                            pitch='0.0', roll='0.0', yaw='%f'%wp_sub.transform.rotation.yaw)

                                route_id += 1
                        else:
                            duplicates +=1
                    
    tree = ET.ElementTree(root)
    
    len_tree = 0
    for _ in tree.iter('route'):
        len_tree += 1
    print(f"Num routes for {args.town}: {len_tree}")

    if args.save_dir is not None and len_tree > 0:
        tree.write(args.save_file, xml_declaration=True, encoding='utf-8', pretty_print=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, help='output folder with routes')
    parser.add_argument('--scenarios_dir', type=str, help='file containing the route waypoints')
    parser.add_argument('--town', type=str, default='all', help='mention single town, else generates for all towns')
    parser.add_argument('--road_type', help='curved/junction')

    args = parser.parse_args()

    if args.town == 'all':
        towns = ALL_TOWNS 
    else:
        towns = [args.town]

    scenario_name = args.scenarios_dir.split('/')[-2]
    print(f"Generating routes for {scenario_name}")
    for town_ in towns:
        args.town = town_
        args.scenarios_file = os.path.join(args.scenarios_dir, town_ + '_' + scenario_name + '.json')
        route_save_dir = os.path.join(args.save_dir, scenario_name)
        if not os.path.exists(route_save_dir):
            os.makedirs(route_save_dir)
        args.save_file = os.path.join(route_save_dir, town_ + '_' + scenario_name + '.xml')
        main(args)
