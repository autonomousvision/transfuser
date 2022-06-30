import argparse
import re
import json
import csv
import os
import sys
from os import walk
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines


parser = argparse.ArgumentParser()
parser.add_argument('--xml', type=str, default='./leaderboard/data/routes/longest_weathers.xml', help='Routes file.')
parser.add_argument('--results', type=str, default='./results/raw/', help='Folder with json files to be parsed')
parser.add_argument('--save_dir', type=str, default='./results/parsed/', help='Directory for saving csvs')
parser.add_argument('--town_maps', type=str, default='./leaderboard/data/town_maps_xodr', help='Directory containing town map images')


towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06']

reference_coord = {'Town01': (-8.22, -8.187), 'Town02': (-13.102, 0.148), 'Town03': (-291.567, 320.126),
                   'Town04': (-518.496, 398.342), 'Town05': (-317.72, 217.554), 'Town06': (-390.685, -160.232)}

scale = {'Town01': (757/410, 636/345), 'Town02': (434/214, 637/314), 'Town03': (651/605, 637/590), 
         'Town04': (708/940, 627/844 ), 'Town05': (784/540, 632/436), 'Town06': (920/1050, 522/570)}

infraction_to_symbol = {"collisions_layout":("#ff0000","."), 
                           "collisions_pedestrian":("#00ff00","."),
                           "collisions_vehicle":("#0000ff","."),
                           "outside_route_lanes":("#00ffff","."),
                           "red_light":("#ffff00","."),
                           "route_dev":("#ff00ff","."),
                           "route_timeout":("#ffffff","."),
                           "stop_infraction":("#777777","."),
                           "vehicle_blocked":("#000000",".")}


def getPixel(coord, town_name):
    x,y = coord
    pix_x = int((x-reference_coord[town_name][0])*scale[town_name][0])
    pix_y = int(-(y-reference_coord[town_name][1])*scale[town_name][1])
        
    if town_name == 'Town03' or town_name == 'Town04':
        pix_y = int(-(-y-reference_coord[town_name][1])*scale[town_name][1])
    if town_name == 'Town01' or town_name == 'Town02' or town_name == 'Town06':
        pix_x = abs(pix_x)
        pix_y = abs(pix_y)
    
    return pix_x, pix_y


def plotPixel(coord, town_name, town_img, color):
    pix_x, pix_y = getPixel(coord, town_name)
    
    length = 6
    width = 3
    town_img[pix_y-length:pix_y+(length+1), pix_x-width:pix_x+(width+1)] = color
    town_img[pix_y-width:pix_y+(width+1), pix_x-length:pix_x+(length+1)] = color
    
    return town_img


def create_legend():
    symbols = [lines.Line2D([], [], color=col, marker=mark,markersize=15) 
               for _,(col,mark) in infraction_to_symbol.items()]
    names = [infraction for infraction,_ in infraction_to_symbol.items()]

    figlegend = plt.figure(figsize=(3,int(0.34*len(names))))
    figlegend.legend(handles=symbols, labels=names)
    figlegend.savefig(os.path.join(args.save_dir, 'legend.png'))

def hex_to_list(hex_str):
    hex_to_dec = {"0":0,"1":1,"2":2,"3":3,"4":4,
                  "5":5,"6":6,"7":7,"8":8,"9":9,
                  "a":10,"b":11,"c":12,"d":13,
                  "e":14,"f":15}
    
    num1 = 16*hex_to_dec[hex_str[1]]+hex_to_dec[hex_str[2]]
    num2 = 16*hex_to_dec[hex_str[3]]+hex_to_dec[hex_str[4]]
    num3 = 16*hex_to_dec[hex_str[5]]+hex_to_dec[hex_str[6]]
    
    return [num1,num2,num3]


def get_infraction_coords(infraction_description):
    combined = re.findall('\(x=.*\)', infraction_description)
    if len(combined)>0:
        coords_str = combined[0][1:-1].split(", ")
        coords = [float(coord[2:]) for coord in coords_str]
    else:
        coords=["-","-","-"]
    
    return coords


def main():
    root = ET.parse(args.xml).getroot()
    
    # build route matching dict
    route_matching = {}
    if ('weather' in [elem.tag for elem in root.iter()]):
        for route,weather_daytime in zip(root.iter('route'),root.iter('weather')):
            combined = re.findall('[A-Z][^A-Z]*', weather_daytime.attrib["id"])
            weather =  "".join(combined[:-1])
            daytime = combined[-1]
            route_matching[route.attrib["id"]]={'town':route.attrib["town"],
                                            'weather': weather,
                                                "daytime": daytime}
    else:
        for route in root.iter('route'):
            route_matching[route.attrib["id"]]={'town':route.attrib["town"],
                                            'weather': "Clear",
                                                "daytime": "Noon"}
    
    #_, _, filenames = next(walk(args.results))
    filenames = []
    for foldername, _, cur_filenames in walk(args.results):
        paths = []
        for filename in cur_filenames:
            if(filename.endswith(('.json'))):
                paths.append(os.path.join(foldername, filename))
        filenames += paths

    # lists to aggregate multiple json files
    route_evaluation = []
    total_score_labels = []
    total_score_values = []

    total_km_driven = 0.0
    total_infractions = {}
    total_infractions_per_km = {}

    abort = False
    # aggregate files
    for f in filenames:
        #with open(os.path.join(args.results,f)) as json_file:
        with open(f) as json_file:
            evaluation_data = json.load(json_file)

            if(len(total_infractions) == 0):
                for infraction_name in evaluation_data['_checkpoint']['global_record']['infractions']:
                    total_infractions[infraction_name] = 0

            for record in evaluation_data['_checkpoint']['records']:
                if(record['scores']['score_route'] <= 0.00000000001 ):
                    print("Warning: There is a route where the agent did not start to drive." + " Route ID: " + record['route_id'], file=sys.stderr)
                if(record['status'] == "Failed - Agent couldn\'t be set up"):
                    print("Error: There is at least one route where the agent could not be set up. Aborting." + " Route ID: " + record['route_id'], file=sys.stderr)
                    abort = True
                if(record['status'] == "Failed"):
                    print("Error: There is at least one route that failed. Aborting." + " Route ID: " + record['route_id'], file=sys.stderr)
                    abort = True
                if(record['status'] == "Failed - Simulation crashed"):
                    print("Error: There is at least one route where the simulation crashed. Aborting." + " Route ID: " + record['route_id'], file=sys.stderr)
                    abort = True

                percentage_of_route_completed = record['scores']['score_route'] / 100.0
                route_length_km = record['meta']['route_length'] / 1000.0
                driven_km = percentage_of_route_completed * route_length_km
                total_km_driven += driven_km

                for infraction_name in evaluation_data['_checkpoint']['global_record']['infractions']:
                    if(infraction_name == 'outside_route_lanes'):
                        if(len(record['infractions'][infraction_name]) > 0):
                            meters_off_road = re.findall("\d+\.\d+", record['infractions'][infraction_name][0])[0]
                            km_off_road = float(meters_off_road) / 1000.0
                            total_infractions[infraction_name] += km_off_road
                    else:
                        num_infraction = len(record['infractions'][infraction_name])
                        total_infractions[infraction_name] += num_infraction


            eval_data = evaluation_data['_checkpoint']['records']
            total_scores = evaluation_data["values"]
            route_evaluation += eval_data
            
            total_score_labels = evaluation_data["labels"]
            total_score_values += [[float(score)*len(eval_data) for score in total_scores]]

    for key in total_infractions:
        total_infractions_per_km[key] = total_infractions[key] / total_km_driven
        if(key == 'outside_route_lanes'):
            total_infractions_per_km[key] = total_infractions_per_km[key] * 100.0 #Since this infraction is a percentage, we put it in rage [0.0, 100.0]

    total_score_values = np.array(total_score_values)

    if((len(route_evaluation) % len(route_matching) != 0)):
        print("Error: The number of completed routes (" + str(len(route_evaluation)) + ") is not a multiple of the total routes (" + str(len(route_matching)) + "). Check if there are missing results. Aborting.", file=sys.stderr)
        abort = True

    if(abort == True):
        exit()

    total_score_values = total_score_values.sum(axis=0)/len(route_evaluation)

    for idx, value in enumerate(total_score_labels):
        if(value == 'Collisions with pedestrians'):
            total_score_values[idx] = total_infractions_per_km['collisions_pedestrian']
        elif (value == 'Collisions with vehicles'):
            total_score_values[idx] = total_infractions_per_km['collisions_vehicle']
        elif (value == 'Collisions with layout'):
            total_score_values[idx] = total_infractions_per_km['collisions_layout']
        elif (value == 'Red lights infractions'):
            total_score_values[idx] = total_infractions_per_km['red_light']
        elif (value == 'Stop sign infractions'):
            total_score_values[idx] = total_infractions_per_km['stop_infraction']
        elif (value == 'Off-road infractions'):
            total_score_values[idx] = total_infractions_per_km['outside_route_lanes']
        elif (value == 'Route deviations'):
            total_score_values[idx] = total_infractions_per_km['route_dev']
        elif (value == 'Route timeouts'):
            total_score_values[idx] = total_infractions_per_km['route_timeout']
        elif (value == 'Agent blocked'):
            total_score_values[idx] = total_infractions_per_km['vehicle_blocked']

    # dict to extract unique identity of route in case of repetitions
    route_to_id = {}
    for route in route_evaluation:
        route_to_id[route["route_id"]] = ''.join(i for i in route["route_id"] if i.isdigit())

    # build table of relevant information
    total_score_info = [{"label":label, "value":value} for label,value in zip(total_score_labels,total_score_values)]
    route_scenarios = [{"route":route["route_id"],
                        "town":route_matching[route_to_id[route["route_id"]]]["town"],
                        "weather": route_matching[route_to_id[route["route_id"]]]["weather"],
                        "daytime": route_matching[route_to_id[route["route_id"]]]["daytime"],
                        "duration": route["meta"]["duration_game"],
                        "length": route["meta"]["route_length"],
                        "score": route["scores"]["score_composed"],
                        "completion":route["scores"]["score_route"],
                        "status": route["status"],
                        "infractions": [(key,
                                        len(item),
                                        [get_infraction_coords(description) for description in item]) 
                                        for key,item in route["infractions"].items()]} 
                    for route in route_evaluation]

    # compute aggregated statistics and table for each filter
    filters = ["route","town","weather","daytime","status"]
    evaluation_filtered = {}

    for filter in filters:
        subcategories = np.unique(np.array([scenario[filter] for scenario in route_scenarios]))
        route_scenarios_per_subcategory = {}
        evaluation_per_subcategory = {}
        for subcategory in subcategories:
            route_scenarios_per_subcategory[subcategory]=[]
            evaluation_per_subcategory[subcategory] = {}
        for scenario in route_scenarios:
            route_scenarios_per_subcategory[scenario[filter]].append(scenario)
        for subcategory in subcategories:
            scores = np.array([scenario["score"] for scenario in route_scenarios_per_subcategory[subcategory]])
            completions = np.array([scenario["completion"] for scenario in route_scenarios_per_subcategory[subcategory]])
            durations = np.array([scenario["duration"] for scenario in route_scenarios_per_subcategory[subcategory]])
            lengths = np.array([scenario["length"] for scenario in route_scenarios_per_subcategory[subcategory]])

            infractions = np.array([[infraction[1] for infraction in scenario["infractions"]] 
                        for scenario in route_scenarios_per_subcategory[subcategory]])

            scores_combined = (scores.mean(),scores.std())
            completions_combined = (completions.mean(),completions.std())
            durations_combined = (durations.mean(), durations.std())
            lengths_combined = (lengths.mean(), lengths.std())
            infractions_combined = [(mean,std) for mean,std in zip(infractions.mean(axis=0),infractions.std(axis=0))]

            evaluation_per_subcategory[subcategory] = {"score":scores_combined,
                                                    "completion": completions_combined,
                                                    "duration":durations_combined,
                                                    "length":lengths_combined,
                                                    "infractions": infractions_combined} 
        evaluation_filtered[filter]=evaluation_per_subcategory

    # write output csv file
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    f = open(os.path.join(args.save_dir, 'results.csv'),'w')      # Make file object first
    csv_writer_object = csv.writer(f)   # Make csv writer object
    # writerow writes one row of data given as list object
    for info in total_score_info:
        #print([info[key] for key in info.keys()])
        csv_writer_object.writerow([item for _,item in info.items()])
    csv_writer_object.writerow([""])

    for filter in filters:
        infractions_types = []
        for infraction in route_scenarios[0]["infractions"]:
            infractions_types.append(infraction[0]+" mean")
            infractions_types.append(infraction[0]+" std")

        # route aggregation table has additional columns
        if filter == "route":
            csv_writer_object.writerow([filter,"town","weather","daytime","score mean","score std","completion mean","completion std","duration mean","duration std","length mean","length std"]+
                                infractions_types)
        else:
            csv_writer_object.writerow([filter,"score mean","score std","completion mean","completion std","duration mean","duration std","length mean","length std"]+
                                infractions_types)
        
        sorted_keys = sorted(evaluation_filtered[filter].keys(), key=lambda x: int(x.split('_')[-1]) if len(x.split('_')) >= 2 else -1)
        #sorted_keys = sorted(evaluation_filtered[filter].keys(), lambda x: -1)
        
        #for key,item in evaluation_filtered[filter].items():
        for key in sorted_keys:
            item = evaluation_filtered[filter][key]
            infractions_output = []
            for infraction in item["infractions"]:
                infractions_output.append(infraction[0])
                infractions_output.append(infraction[1])
            if filter == "route":
                csv_writer_object.writerow([key,
                                            route_matching[route_to_id[key]]["town"],
                                            route_matching[route_to_id[key]]["weather"],
                                            route_matching[route_to_id[key]]["daytime"],
                                            item["score"][0],item["score"][1],
                                            item["completion"][0],item["completion"][1],
                                            item["duration"][0],item["duration"][1],
                                            item["length"][0],item["length"][1]]+
                                            infractions_output)
            else:
                csv_writer_object.writerow([key,
                                            item["score"][0],item["score"][1],
                                            item["completion"][0],item["completion"][1],
                                            item["duration"][0],item["duration"][1],
                                            item["length"][0],item["length"][1]]+
                                            infractions_output)
        csv_writer_object.writerow([""])
        
    csv_writer_object.writerow(["town","weather","daylight","infraction type","x","y","z"])
    # writerow writes one row of data given as list object
    for scenario in route_scenarios:
        #print([scenario[key] for key in scenario.keys() if key!="town"])
        for infraction in scenario["infractions"]:
            for coord in infraction[2]:
                if type(coord[0]) != str:
                    csv_writer_object.writerow([scenario["town"],scenario["weather"],scenario["daytime"],
                                                infraction[0]]+coord)
    csv_writer_object.writerow([""])
    f.close()

    # load town maps for plotting infractions
    town_maps = {}
    for town_name in towns:
        town_maps[town_name] = np.array(Image.open(os.path.join(args.town_maps, town_name+'.png')))[:,:,:3]

    create_legend()

    for scenario in route_scenarios:
        for infraction in scenario["infractions"]:
            for coord in infraction[2]:
                if type(coord[0]) != str:
                    
                    x = coord[0]
                    y = coord[1]
                    
                    town_name = scenario["town"]
                    
                    hex_str,_ = infraction_to_symbol[infraction[0]]
                    color = hex_to_list(hex_str)
                    # plot infractions
                    town_maps[town_name] = plotPixel((x,y), town_name, town_maps[town_name], color)

    for town_name in towns:
        tmap = Image.fromarray(town_maps[town_name])
        tmap.save(os.path.join(args.save_dir, town_name+'.png'))


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    main()

