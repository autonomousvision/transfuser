import os
import json
import argparse
import numpy as np
import carla

from utils import interpolate_trajectory, gen_skeleton_dict, gen_scenarios

PRECISION = 2
SAMPLING_DISTANCE = 30


def generate_scenario_10(carla_map, world=None):

    carla_topology = carla_map.get_topology()
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    other_actors = []
    count_all_routes  = 0
    duplicates = 0
    trigger_points = []
    
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
                                wp_p = wp_p.previous(PRECISION)[0] # multiple routes are possible, we are only considering one possiblity
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
                                wp_n = wp_n.next(PRECISION)[0] # multiple routes are possible, we are only considering one possiblity
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
                                trigger_points +=  [extended_route[0][0]]

                            else:
                                duplicates +=1

            other_actors = [None] * len(trigger_points)

    return trigger_points, other_actors


FUNC_SCENARIO_TYPE =  {'Scenario10': generate_scenario_10}


def town_scenario_tp_gen(town_, carla_map , scen_type, save_dir, world):

    scen_save_dir = os.path.join(save_dir, scen_type)
    if not os.path.exists(scen_save_dir):
        os.makedirs(scen_save_dir)

    ego_triggers, other_triggers = FUNC_SCENARIO_TYPE[scen_type](carla_map, world)
    town_x_scen_y_dict = gen_skeleton_dict([town_], [scen_type])
    
    trigger_point_dict_lst = []
    for ego_trig, oa_trig in zip(ego_triggers,other_triggers):
        trigger_pts_dict = {}
        
        if ego_trig:
            ego_trig_dict = {"pitch": ego_trig.rotation.pitch, "yaw": ego_trig.rotation.yaw, "x": ego_trig.location.x, "y":ego_trig.location.y, "z": ego_trig.location.z }
            trigger_pts_dict["transform"] = ego_trig_dict
            
        trigger_pts_dict["other_actors"] = oa_trig
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