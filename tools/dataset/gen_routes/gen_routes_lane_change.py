import os
import random
import lxml.etree as ET
import argparse

import carla

from utils import ALL_TOWNS, interpolate_trajectory

PRECISION = 10 
DISTANCE = 380
PRUNE_ROUTES_MIN_LEN = 20
LIMIT_FINAL_ROUTES = 50

def get_possible_lane_changes(current_waypoint):
    
    all_lefts =  {} 
    all_rights = {}
    tmp_wp = current_waypoint

    # check number of left lanes available
    lane_side = 'l'
    while True:
        left_w = tmp_wp.get_left_lane()
        if  left_w and left_w.lane_type == carla.LaneType.Driving and  0 <= abs(left_w.transform.rotation.yaw - tmp_wp.transform.rotation.yaw) <= 10 :
            all_lefts[lane_side] = left_w
            tmp_wp  = left_w 
            lane_side += 'l'
        else:
            break

    tmp_wp = current_waypoint

    # check number of right lanes available
    lane_side = 'r'
    while True:
        right_w = tmp_wp.get_right_lane()
        if right_w and right_w.lane_type == carla.LaneType.Driving and 0 <= abs(right_w.transform.rotation.yaw - tmp_wp.transform.rotation.yaw) <= 10:
            all_rights[lane_side] = right_w
            tmp_wp = right_w
            lane_side += 'r'
        else:
            break
        
    current_dict = {'n' : current_waypoint}
    all_choices = {**all_lefts, **all_rights, **current_dict}
    return all_choices  


def main(args):
    #initialize
    route_id = 0
    duplicate_list = []
    count_all_routes  = 0
    duplicates = 0
    distance_ = 100
    wp_length = 9
    PRECISION_small = 1
    WP_extended = 150
    
    root = {}
    root['lr'] = ET.Element('routes')
    root['ll'] = ET.Element('routes')
    root['rr'] = ET.Element('routes')
    root['rl'] = ET.Element('routes')

    final_save_dirs = {}
    # create sub-directories for each lane change
    for key_, _ in root.items():
        sub_path = os.path.join(args.save_dir, key_)
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)
        final_save_dirs[key_] = sub_path

    # set up carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    world = client.load_world(args.town)
    carla_map = world.get_map()
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    
    print(f"Num waypoints for {args.town}: {len(topology)}")

    # iterate over each waypoint and check if lange changes can be constructed
    for wp_it, cur_wp in enumerate(topology):
        wp_list_nxt = [cur_wp]

        if not cur_wp.is_junction:    
            tmp_distance_= 0

            # find the forward waypoints from current
            # add waypoints until junction appears or crosses threshold distance
            while(True):
                cur_wp_ = wp_list_nxt[-1]
                try:
                    nxt_wp = cur_wp_.next(PRECISION)[0]
                except:
                    break
                if not nxt_wp.is_junction and tmp_distance_ < distance_ :
                    wp_list_nxt.append(nxt_wp)
                    tmp_distance_ += PRECISION
                else:
                    break

        if len(wp_list_nxt) > wp_length:
            final_wps_list =   wp_list_nxt
            end_point = final_wps_list[-1]
            mid_point = final_wps_list[int(len(final_wps_list)/2)]
              
            try:
                all_choices_ep = get_possible_lane_changes(end_point)
                all_choices_mp = get_possible_lane_changes(mid_point)
                if not len(all_choices_ep) > 1 and not len(all_choices_mp) > 1:
                    continue                
            except:
                continue
            
            all_combs_split = {'lr':[], 'll':[], 'rr':[], 'rl':[]}
            all_combs = []
            for key_ep, ep_ in all_choices_ep.items():
                for key_mp, mp_ in all_choices_mp.items():
                    if key_ep != key_mp:
                        if key_mp != 'n':
                            mp_direction_ = set(key_mp)#[0]
                            mp_cnt_ = len(key_mp)
                            
                            ep_direction_ = set(key_ep)
                            ep_cnt_ = len(key_ep)
                            
                            if mp_direction_ == {'l'}:
                                if ep_direction_ == {'r'} or ep_direction_ == {'n'}:
                                    lane_change_key = 'lr'
                                elif ep_direction_ == {'l'}:
                                    if mp_cnt_ > ep_cnt_:
                                        lane_change_key = 'lr'
                                    else:
                                        lane_change_key = 'll'
                                        
                            elif mp_direction_ =={'r'}:
                                if ep_direction_ == {'l'} or ep_direction_ == {'n'}:
                                    lane_change_key = 'rl'
                                elif ep_direction_ == {'r'}:
                                    if mp_cnt_ > ep_cnt_:
                                        lane_change_key = 'rl'
                                    else:
                                        lane_change_key = 'rr'

                            final_wps = [final_wps_list[0], mp_, ep_]
                            all_combs_split[lane_change_key].append(final_wps) 
                                        
            truncated_wp_lst = [final_wps_list]
                                
            locations = []
            for wps_sub in truncated_wp_lst:
                    locations.append((wps_sub[0].transform.location.x, wps_sub[0].transform.location.y, wps_sub[-1].transform.location.x, wps_sub[-1].transform.location.y, wps_sub[0].transform.rotation.yaw))
        
            for location_ in locations:

                # remove routes that are duplicates or in the next lane
                flag_cum_ctr = []
                for loc_dp in duplicate_list:
                    flag_ctrs = [True if prev_loc - PRECISION_small <= curr_loc <= prev_loc+ PRECISION_small else False for curr_loc, prev_loc in zip(location_, loc_dp) ] # threshold hardset
                    flag_AND_ctr = all(flag_ctrs)
                    flag_cum_ctr.append(flag_AND_ctr)
                is_loc_dup = any(flag_cum_ctr)

                # add the route if it does not exist already
                if not is_loc_dup:
                    duplicate_list.append(locations[0])

                    for all_combs_key, all_combs in all_combs_split.items():
                        for j_, wps_ in enumerate(all_combs):
                                count_all_routes +=1
                                
                                wps_tmp =  [wps_[0].transform.location, wps_[1].transform.location , wps_[-1].transform.location]
                                try:
                                    extended_route = interpolate_trajectory(carla_map, wps_tmp)
                                    
                                    # add route if its length constraints satisfy these constraints
                                    #for generating more routes change the lengths
                                    if len(extended_route) >  WP_extended or len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                                        continue
                                except:
                                    continue

                                wps_tmp2 = wps_
                                route = ET.SubElement(root[all_combs_key], 'route', id='%d'%route_id, town=args.town)
                                for k_, wp_sub in enumerate(wps_tmp2):
                                    ET.SubElement(route, 'waypoint', x='%f'%(wp_sub.transform.location.x), y='%f'%(wp_sub.transform.location.y), z='%f'%(wp_sub.transform.location.z), 
                                                                pitch='0.0', roll='0.0', yaw='%f'%wp_sub.transform.rotation.yaw)

                                    route_id += 1
                else:
                    duplicates +=1
    
    # save to file
    tree = {}
    root_pruned = {}
    for key_ in ['rr', 'lr', 'll', 'rl']:      

        root_pruned[key_] = ET.Element('routes')
        index_list = list(range(len(root[key_])))
        random.shuffle(index_list)
        index_list_pruned = index_list[:LIMIT_FINAL_ROUTES]
        
        route_id_pruned = 0
        
        for ind_, child_ in enumerate(root[key_]):
            if ind_ in index_list_pruned:
                route_new = ET.SubElement(root_pruned[key_], 'route', id='%d'%route_id_pruned, town=args.town)
                
                for subelement_ in child_.findall('waypoint'):
                    ET.SubElement(route_new, subelement_.tag, subelement_.attrib)
                route_id_pruned +=1

        tree = ET.ElementTree(root_pruned[key_])
        
        len_tree = 0
        for _ in tree.iter('route'):
            len_tree += 1
        print(f"Num routes for {args.town}: {len_tree}")

        if args.save_dir is not None and len_tree > 0:
            filename_ = os.path.join(final_save_dirs[key_], town_ + '_' + key_ + '.xml')
            tree.write(filename_, xml_declaration=True, encoding='utf-8', pretty_print=True)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, help='output folder with routes')
    parser.add_argument('--town', type=str, default='all', help='mention single town, else generates for all towns')

    args = parser.parse_args()

    if args.town == 'all':
        towns = ALL_TOWNS 
    else:
        towns = [args.town]

    for town_ in towns:
        args.town = town_
        main(args)