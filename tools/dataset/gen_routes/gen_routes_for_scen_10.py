import os
import argparse
import lxml.etree as ET

import numpy as np
import carla

from utils import ALL_TOWNS, \
        parse_annotations_file, interpolate_trajectory, scan_route_for_scenarios

ID_START = 0
PRECISION = 2
SAMPLING_DISTANCE = 30

def main(args):
    
    scenarios_list = parse_annotations_file(args.scenarios_file)
    route_id = ID_START
    root = ET.Element('routes')

    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()

    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    count_all_routes  = 0
    duplicates = 0
    
    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print ('got %d traffic lights'%len(traffic_lights_list))

    # group traffic lights and associate them into one group
    junction_bbs_centers = []
    junctions_ = []
    duplicate_list = []
    for wp_it, waypoint in enumerate(topology):
        if waypoint.is_junction:
            junc_ = waypoint.get_junction()
            jbb_ = junc_.bounding_box
            jbb_center = [round(jbb_.location.x, 2), round(jbb_.location.y, 2)]
            if jbb_center not in junction_bbs_centers:
                
                junction_bbs_centers.append(jbb_center)
                junctions_.append(junc_)

    pole_ind = []
    grps_  = []
    
    for tl_ in traffic_lights_list:
        grp_tl = tl_.get_group_traffic_lights()
        grp_tl_locs = [ ( round(gtl_.get_transform().location.x, 2), round(gtl_.get_transform().location.y,2)) for gtl_ in grp_tl ]
        pole_ind.append(tl_.get_pole_index())
        if grp_tl_locs not in grps_:
            grps_.append(grp_tl_locs)
    
    bb_flags = []
    for grp_ in grps_:
        midpt_grp = [sum(i)/len(grp_) for i in zip(*grp_)]
        
        # find the associated bounding box
        # ASSUMPTION every traffic light is associated to the junctions
        # Usually it is around ~5m from the midpoint of the group
        
        grp_bb_dist = [np.sqrt((bb_c[0]-midpt_grp[0])**2 + (bb_c[1]-midpt_grp[1])**2) for bb_c in junction_bbs_centers]
        bb_dist, bb_idx = min((val, idx) for (idx, val) in enumerate(grp_bb_dist))
        bb_flags.append(bb_idx)
    
    unsignalized_junctions = [junc_ for i, junc_ in enumerate(junctions_) if i not in bb_flags]
    unsignalized_junctions_bbs = [(junc_.bounding_box.location.x, junc_.bounding_box.location.y) for i, junc_ in enumerate(junctions_) if i not in bb_flags]
    if len(unsignalized_junctions):
        count_all_routes  = 0
        duplicates = 0
        PRECISION = 2
        SAMPLING_DISTANCE = 30
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = (junc_.bounding_box.location.x , junc_.bounding_box.location.y)
                if jbb_ in unsignalized_junctions_bbs:
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
                                potential_scenarios_definitions, _ = scan_route_for_scenarios(carla_map.name, extended_route, scenarios_list)
                                
                                wps_ = [wps_[0], wps_[-1]]                   

                                if (len(potential_scenarios_definitions) > 0):
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