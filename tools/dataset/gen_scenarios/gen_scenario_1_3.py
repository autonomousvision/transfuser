import os
import json
import argparse
import math
from utils import interpolate_trajectory, gen_skeleton_dict, gen_scenarios

DOT_PROD_SLACK = 0.02
PRECISION = 2
DISTANCE = 380 # length in meters
NUM_WAYPOINTS_DISTANCE = int(DISTANCE / PRECISION)
PRUNE_ROUTES_MIN_LEN = 20 #set this to prune shorter routes
MIN_ROUTE_LENGTH = 4


def generate_scenario_3(carla_map):
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    other_actors = []
    count_all_routes  = 0
    trigger_points = []
    duplicate_list = []
    
    for wp_it, waypoint in enumerate(topology):
        cur_wp = waypoint

        wp_list_nxt = [cur_wp]
        if not cur_wp.is_junction:    
            
            #forward wp
            while(True):
                cur_wp_ = wp_list_nxt[-1]
                try:
                    # THIS ONLY CONSIDERS ONE ROUTE
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
                    # THIS ONLY CONSIDERS ONE ROUTE
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
                count_all_routes +=1
                for i_ in range(len(final_wps_list)):
                    
                    tmp_wps =  final_wps_list[i_*NUM_WAYPOINTS_DISTANCE : i_*NUM_WAYPOINTS_DISTANCE + NUM_WAYPOINTS_DISTANCE ]
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
                        
                        if not dot_ < -1 +  DOT_PROD_SLACK:#dot_ > -1 - DOT_PROD_SLACK 
                            truncated_wp_lst.append(tmp_wps) 


                    #can make this as a function
                    locations =[]
                    for wps_sub in truncated_wp_lst:
                        # for wp_ in wps_sub:
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
                            
                            duplicate_list.append(locations[j_])
                            wps_tmp =  [wps_[0].transform.location, wps_[-1].transform.location]
                            
                            # this might also lead to a different route
                            extended_route = interpolate_trajectory(carla_map, wps_tmp)
                            
                            if len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                                continue
                            else:
                                # each extended element is of type (Transform, LaneFollow)
                                # route is with hop resolution 1.0 so the mid point is the mid element
                                trigger_point = extended_route[int(len(extended_route)/2)][0]
                                trigger_points +=  [trigger_point]
                            
    other_actors = [None] * len(trigger_points)
    return trigger_points, other_actors


def generate_scenario_1(carla_map):
    
    # carla get waypoints
    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    # initializations and some constants iused
    other_actors = []
    count_all_routes  = 0
    trigger_points = []
    duplicate_list = []
    
    for wp_it, waypoint in enumerate(topology):

        cur_wp = waypoint
        wp_list_nxt = [cur_wp]
        if not cur_wp.is_junction:    
            
            #forward wp
            while(True):
                cur_wp_ = wp_list_nxt[-1]
                try:
                    # THIS ONLY CONSIDERS ONE ROUTE
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
                    # THIS ONLY CONSIDERS ONE ROUTE
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
                count_all_routes +=1
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
                        
                        if not dot_ < -1 +  DOT_PROD_SLACK:#dot_ > -1 - DOT_PROD_SLACK 
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
                            duplicate_list.append(locations[j_])
                            wps_tmp =  [wps_[0].transform.location, wps_[-1].transform.location]
                            
                            # this might also lead to a different route
                            extended_route = interpolate_trajectory(carla_map, wps_tmp)
                            
                            if len(extended_route) < PRUNE_ROUTES_MIN_LEN:
                                continue
                            else:
                                # each extended element is of type (Transform, LaneFollow)
                                # route is with hop resolution 1.0 so the mid point is the mid element
                                trigger_point = extended_route[int(len(extended_route)/2)][0]
                                trigger_points +=  [trigger_point]
                                
                                                            
    other_actors = [None] * len(trigger_points)
    return trigger_points, other_actors


FUNC_SCENARIO_TYPE =  {'Scenario1': generate_scenario_1, 'Scenario3': generate_scenario_3 }


def town_scenario_tp_gen(town_, carla_map , scen_type, save_dir, world):
    
    scen_save_dir = os.path.join(save_dir, scen_type)
    if not os.path.exists(scen_save_dir):
        os.makedirs(scen_save_dir)

    ego_triggers, other_triggers = FUNC_SCENARIO_TYPE[scen_type](carla_map)
    town_x_scen_y_dict = gen_skeleton_dict([town_], [scen_type])

    trigger_point_dict_lst = []
    for ego_trig, oa_trig in zip(ego_triggers,other_triggers):
        trigger_pts_dict = {}
        
        if ego_trig:
            ego_trig_dict = {"pitch": ego_trig.rotation.pitch, "yaw": ego_trig.rotation.yaw, "x": ego_trig.location.x, "y":ego_trig.location.y, "z": ego_trig.location.z }
            trigger_pts_dict["transform"] = ego_trig_dict
        if oa_trig:
            oa_trig_dict = {"pitch": ego_trig.rotation.pitch, "yaw": ego_trig.rotation.yaw, "x": ego_trig.location.x, "y":ego_trig.location.y, "z": ego_trig.location.z }
            trigger_pts_dict["other_actors"] = oa_trig_dict
        trigger_point_dict_lst.append(trigger_pts_dict)
    
    for i, town_dict_ in enumerate(town_x_scen_y_dict["available_scenarios"]):
        if town_ in  town_dict_.keys():
            for j, scens_dict in enumerate(town_dict_[town_]):
                if scens_dict['scenario_type'] == scen_type:
                    scens_dict['available_event_configurations'] = trigger_point_dict_lst
        
    print(f"Num trigger points for {town_} {scen_type}: {len(trigger_point_dict_lst)}")
    
    with open(os.path.join(scen_save_dir, f"{town_}_{scen_type}.json"), 'w') as f:
        json.dump(town_x_scen_y_dict, f, indent=2, sort_keys=True)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--towns', type=str, default='all', help='mention single town, else generates for all towns')
    parser.add_argument('--save_dir', type=str, help='output folder with scenarios')
    
    args = parser.parse_args()
    gen_scenarios(args, FUNC_SCENARIO_TYPE, town_scenario_tp_gen)