import os
import sys
import numpy as np
import json
import argparse
import xml.etree.ElementTree as ET
import cv2
import copy
import glob
import ast
from  pathlib import Path


def main(args):
	if args.towns == 'all':
		towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']
	else:
		towns = [args.towns]

	# aggregate transforms of all the scenarios present in the provided json and save them individually
	scenarios = open(args.scenarios_file, 'r')
	scenarios_data = json.load(scenarios)
	scenarios.close()

	available_event_configurations = []
	scenario_types = []
	available_scenarios = scenarios_data['available_scenarios'][0]
	for town in available_scenarios.keys():
	    if town not in towns:
	        continue
	    scenarios_town = available_scenarios[town]
	    for scenario in scenarios_town:
	        transforms_list = scenario['available_event_configurations']
	        scenario_type = scenario['scenario_type']
	        for transform in transforms_list:
	            curr_transform = transform['transform']
	            for attrib in curr_transform.keys():
	                curr_transform[attrib] = float(curr_transform[attrib])
	            curr_transform['z'] = 0.0
	            if curr_transform not in available_event_configurations:
	                available_event_configurations.append(curr_transform)
	        scenario_types.append(scenario_type)

	available_event_configurations_dict_list = []
	for sample in available_event_configurations:
	    available_event_configurations_dict_list.append({'transform': sample})
	print (len(available_event_configurations_dict_list))

	# sample dense points along each trigger transform provided
	aggr_available_event_config = []
	for curr_transform in available_event_configurations_dict_list:
	    augmented_transforms = []
	    def add_yaw_augmentation(transform): # trigger points can have any orientation
	        aggr_transform = []
	        for mul in range(4):
	            aug_transform = copy.deepcopy(transform)
	            aug_transform['transform']['yaw'] += mul*90.0
	            aggr_transform.append(aug_transform)
	        return aggr_transform
	    
	    augmented_transforms += add_yaw_augmentation(curr_transform) # add base transform
	    for dist in range(-10,11,5): # sample extra points along each axes
	        if dist == 0: # base transform already added
	            continue
	        new_transform_x = copy.deepcopy(curr_transform)
	        new_transform_x['transform']['x'] += dist
	        augmented_transforms += add_yaw_augmentation(new_transform_x)
	        new_transform_y = copy.deepcopy(curr_transform)
	        new_transform_y['transform']['y'] += dist
	        augmented_transforms += add_yaw_augmentation(new_transform_y)
	    
	    aggr_available_event_config += augmented_transforms
	print (len(aggr_available_event_config))

	json_list_data = {'available_scenarios': []}
	for town in towns:
	    all_scenarios = []
	    for index in range(1,11):
	        if index == 2 or index == 5 or index == 6: # discard lane changing scenarios since expert is not configured for it
	            continue
	        all_scenarios.append({'available_event_configurations': aggr_available_event_config, 'scenario_type': 'Scenario%d'%index})
	    json_list_data['available_scenarios'].append({town: all_scenarios})

	with open(args.save_file, 'w') as f:
	    json.dump(json_list_data, f, indent=4)
	f.close()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--scenarios_file', type=str, required=True, help='original file reference scenarios')
	parser.add_argument('--save_file', type=str, required=False, help='output file with dense scenarios')
	parser.add_argument('--towns', type=str, required=True, help='Town01/Town02/Town03/Town04/Town05/Town06/all')
	
	args = parser.parse_args()

	main(args)