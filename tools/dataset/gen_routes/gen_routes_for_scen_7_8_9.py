import os
import argparse
import lxml.etree as ET

import math
import numpy as np
import carla

from utils import ALL_TOWNS, \
        parse_annotations_file, interpolate_trajectory, scan_route_for_scenarios

ID_START = 0
SAMPLING_DISTANCE = 30
PRECISION = 2


def main(args):
    
    route_id = ID_START

    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()
    actors = world.get_actors()
    traffic_lights_list = actors.filter('*traffic_light')
    print ('got %d traffic lights'%len(traffic_lights_list))

    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    root_7 = ET.Element('routes')
    root_8 = ET.Element('routes')
    root_9 = ET.Element('routes')
    roots_ = {'Scenario7': root_7 , 'Scenario8':root_8, 'Scenario9':root_9}
    count_all_routes  = 0
    duplicates = 0

    scenarios_list= {}
    scenarios_list['Scenario7'] = parse_annotations_file(args.scenarios_file['Scenario7'])
    scenarios_list['Scenario8'] = parse_annotations_file(args.scenarios_file['Scenario8'])
    scenarios_list['Scenario9'] = parse_annotations_file(args.scenarios_file['Scenario9'])

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
                    
        
    signalized_junctions = [junc_ for i, junc_ in enumerate(junctions_) if i in bb_flags]
    signalized_junctions_bbs = [(junc_.bounding_box.location.x, junc_.bounding_box.location.y) for i, junc_ in enumerate(junctions_) if i  in bb_flags]
    
    if len(signalized_junctions):

        count_all_routes  = 0
        duplicates = 0
        duplicate_list = []
        for wp_it, waypoint in enumerate(topology):
            if waypoint.is_junction:
                junc_ = waypoint.get_junction()
                jbb_ = (junc_.bounding_box.location.x , junc_.bounding_box.location.y)
                if  jbb_ in signalized_junctions_bbs:
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
                        # root selection
                        
                        cur_wp = final_wps_list[int(len(final_wps_list)/2)]
                        prev_wp = final_wps_list[0]
                        nxt_wp = final_wps_list[-1]

                        vec_wp_nxt = cur_wp.transform.location - nxt_wp.transform.location
                        vec_wp_prev = cur_wp.transform.location -  prev_wp.transform.location

                        dot_ = ( vec_wp_nxt.x * vec_wp_prev.x)  + (vec_wp_prev.y * vec_wp_nxt.y)
                        det_ = ( vec_wp_nxt.x * vec_wp_prev.y) -  (vec_wp_prev.x * vec_wp_nxt.y)
                        
                        angle_bet = math.atan2(det_,dot_)
                        
                        if angle_bet < 0 :
                            angle_bet += 2 * math.pi
                            
                        angle_deg =  angle_bet * 180 / math.pi
                        
                        if 160 < angle_deg < 195:
                            # scenario 7: 
                            key_ = 'Scenario7'
                            root = roots_['Scenario7']

                        elif 10 < angle_deg < 160:
                            # scenario 8: 
                            key_ = 'Scenario9'
                            root = roots_['Scenario9']

                        elif 195 < angle_deg < 350:
                            # scenario 9: 
                            key_ = 'Scenario8'
                            root = roots_['Scenario8']
                        
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
                                potential_scenarios_definitions, _ = scan_route_for_scenarios(carla_map.name, extended_route, scenarios_list[key_])
                                potential_scenarios_definitions = []

                                wps_ = [wps_[0], wps_[-1]]                   

                                if  True or (len(potential_scenarios_definitions) > 0):
                                    route = ET.SubElement(root, 'route', id='%d'%route_id, town=args.town)
                                    
                                    for k_, wp_sub in enumerate(wps_):
                                        ET.SubElement(route, 'waypoint', x='%f'%(wp_sub.transform.location.x), y='%f'%(wp_sub.transform.location.y), z='0.0', 
                                                                pitch='0.0', roll='0.0', yaw='%f'%wp_sub.transform.rotation.yaw)

                                    route_id += 1
                                    
                            else:
                                duplicates +=1
                        
        tree_7 = ET.ElementTree(root_7)
        tree_8 = ET.ElementTree(root_8)
        tree_9 = ET.ElementTree(root_9)

        len_tree = 0
        for _ in tree_7.iter('route'):
            len_tree += 1
        print(f"Num routes for Scenario 7 for {args.town}: {len_tree}")

        if args.save_dir is not None and len_tree > 0:
            tree_7.write(args.save_file['Scenario7'], xml_declaration=True, encoding='utf-8', pretty_print=True)

        len_tree = 0
        for _ in tree_8.iter('route'):
            len_tree += 1
        print(f"Num routes for Scenario 8 for {args.town}: {len_tree}")

        if args.save_dir is not None and len_tree > 0:
            tree_8.write(args.save_file['Scenario8'], xml_declaration=True, encoding='utf-8', pretty_print=True)
        
        len_tree = 0
        for _ in tree_9.iter('route'):
            len_tree += 1
        print(f"Num routes for Scenario 9 for {args.town}: {len_tree}")

        if args.save_dir is not None and len_tree > 0:
            tree_9.write(args.save_file['Scenario9'], xml_declaration=True, encoding='utf-8', pretty_print=True)

        
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

    for town_ in towns:
        args.town = town_
        args.scenarios_file= {}
        args.save_file= {}
        for sceneraio_ in ['Scenario7', 'Scenario8', 'Scenario9']:
            args.scenarios_file[sceneraio_] = os.path.join(args.scenarios_dir, sceneraio_,  town_ + '_' + sceneraio_ + '.json')
            route_save_dir = os.path.join(args.save_dir, sceneraio_)
            if not os.path.exists(route_save_dir):
                os.makedirs(route_save_dir)
            args.save_file[sceneraio_] = os.path.join(route_save_dir, town_ + '_' + sceneraio_ + '.xml')
        main(args)