import os
import random
import json
import argparse
import xml.etree.ElementTree as ET

import math
import numpy as np

import carla
import pygame
pygame.init()

# constants used for visualization
PIXELS_PER_METER = 8
MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0
PIXELS_AHEAD_VEHICLE = 256


class MapImage(object):
    """
    Class encharged of rendering a 2D image from top view of a carla world. Please note that a cache system is used, so if the OpenDrive content
    of a Carla town has not changed, it will read and use the stored image if it was rendered in a previous execution
    """

    def __init__(self, carla_map_name, pixels_per_meter, map_dir,  visualize=True):
        """
        Renders the map image generated based on the world, its map and additional flags that provide extra information about the road network
        """
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0
        self.visualize = visualize
        self.map_dir = map_dir
        self.carla_map_name = carla_map_name

        with open(os.path.join(self.map_dir, f'{carla_map_name}_details.json'), 'r') as f:
            data_ = json.load(f)
        self._world_offset = data_['world_offset']

        self.viz_surface = None
        if self.visualize:
            # load image
            filename = carla_map_name + "_.tga"
            self.viz_surface = pygame.image.load(os.path.join(self.map_dir, filename))

    def create_crop(self, surface, point_original, path_=None, np_array=False):
        
        point_ = self.world_to_pixel(point_original.location)

        temp_surface = pygame.Surface.copy(surface)
        if 'visualize' in path_:
            color_ = pygame.Color(0,255,0)
            self.draw_queried_points(temp_surface, 
                                    self.world_to_pixel,
                                        [point_original], colors=[color_]) 

        rz = pygame.transform.rotozoom

        scaled_original_size = 500 * (1.0 / 0.2)
        hero_map_surface = pygame.Surface((scaled_original_size, scaled_original_size))
        angle =  point_original.theta + np.pi/2 
        offset = [0, 0]
        hero_location_screen = point_
        hero_front = carla.Location(x=np.cos(point_original.theta), y=np.sin(point_original.theta))
        offset[0] += hero_location_screen[0] - hero_map_surface.get_width() / 2
        offset[0] += hero_front.x * PIXELS_AHEAD_VEHICLE
        offset[1] += hero_location_screen[1] - hero_map_surface.get_height() / 2
        offset[1] += hero_front.y * PIXELS_AHEAD_VEHICLE
        forward_pt = [hero_location_screen[0] + hero_front.x * PIXELS_AHEAD_VEHICLE, hero_location_screen[1] + hero_front.y * PIXELS_AHEAD_VEHICLE]

        if 'visualize' in path_:
            self.draw_points(temp_surface, forward_pt)
        
        pt_center = [forward_pt[0] - offset[0], forward_pt[1] - offset[1]]
        hero_map_surface.blit(temp_surface, (-offset[0], -offset[1]))
        
        rotated_map_surface = rz(hero_map_surface, np.degrees(angle), 1)
        top_left_in_pixels_final = [pt_center[0] - 128, pt_center[1] ]
        
        center = (scaled_original_size / 2, scaled_original_size / 2)
        rotation_map_pivot = rotated_map_surface.get_rect(center=center)
        
        window_map_surface = pygame.Surface((scaled_original_size, scaled_original_size))
        window_map_surface.blit(rotated_map_surface, rotation_map_pivot)
        
        final_surface = pygame.Surface((256, 256), flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
        final_surface.blit(window_map_surface, (0,0),(top_left_in_pixels_final[0], top_left_in_pixels_final[1], 256, 256) )
        if path_:
            pygame.image.save(final_surface, path_+'.tga')
            
        if np_array:
            # np array conversion does not swap the axes
            np_surface = pygame.surfarray.array3d(final_surface)
            return np_surface
        else:
            # np array if directly called
            return final_surface

    def create_semlabels_offline(self, surface, points_, paths, suffix, save_dir):
        for i, (point_original, path_) in enumerate(zip(points_, paths)):
            crp_dir = os.path.join(save_dir, os.path.dirname(path_), suffix)
            if not os.path.exists(crp_dir):
                os.makedirs(crp_dir)
            _, crop_name = os.path.split(path_)
            crop_pth = os.path.join(crp_dir, crop_name)
            self.create_crop(surface, point_original, path_=crop_pth)

    def plot_points_nocrop_xml(self, surface, points_, filename, save_dir):
        temp_surface = pygame.Surface.copy(surface)
        rgbs_ = []
        for j , pth_ in enumerate(points_):
            print(f' Plotted route {j+1}/{len(points_)}')
            rgb = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            rgbs_.append(rgb)
            color_pg =  (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

            #draw points
            for i, (point_original) in enumerate(pth_[:-1]):
                sp_, ep_ = pth_[i], pth_[i+1]
                
                # draw start point and end point
                self.draw_queried_points(temp_surface,  self.world_to_pixel, [point_original], colors=[color_pg])

                #draw arrow between sp and ep
                color_white = pygame.Color(255,255,255)
                pygame.draw.lines(temp_surface, color_white, False, [self.world_to_pixel(x.location) for x in [sp_, ep_]], 4)

        pygame.image.save_extended(temp_surface, os.path.join(save_dir, f'{filename}.png'))

    def plot_points_nocrop_json(self, surface, points_, filename, save_dir):
        temp_surface = pygame.Surface.copy(surface)
        rgbs_ = []
        for j , pth_ in enumerate(points_):
            rgb = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
            rgbs_.append(rgb)
            color_pg =  (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

            for i, (point_original) in enumerate(pth_):
                self.draw_queried_points(temp_surface, self.world_to_pixel, [point_original['ego']], colors=[color_pg]) 
            
        pygame.image.save_extended(temp_surface, os.path.join(save_dir, f'{filename}.png'))
        
    def draw_points(self, surface, transform, color=None):
        """ Draws an arrow with a specified color given a transform"""
        end = transform
        start = end
        
        color_tmp = [int(255) for c_ in range(3)] if not color  else color
        color = pygame.Color(*color_tmp) if color is None else color

        pygame.draw.circle(surface, color, start, 5)

    def draw_queried_points(self, map_surface, world_to_pixel , points=[], colors=[]):
        
        def draw_points_local(surface, transform, color=None):
            """ Draws an arrow with a specified color given a transform"""
            start = transform.location
            pygame.draw.circle(surface, color, world_to_pixel(start),18)

        def draw_arrow(surface, transform, color=pygame.Color(193, 125, 17)):
            """ Draws an arrow with a specified color given a transform"""
            begin = carla.Location(x=transform.location.x, y=transform.location.y)
            
            angle = math.radians(transform.theta)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            forward = carla.Location(x=np.cos(angle), y=np.sin(angle))
                        
            end = carla.Location(x=transform.location.x,y=transform.location.y )
            start =  end +  forward
            pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [start, end]], 4)
       
        if len(points):
            for p_, color_  in zip(points,colors):
                draw_points_local(map_surface, p_, color=color_)
                draw_arrow(map_surface, p_)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Converts the world coordinates to pixel coordinates"""
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        """Scales the map surface"""
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))


class Location(object):
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]


class Point(object):
    # add other transforms
    def __init__(self, point):
        self.location = Location(point[:2])
        self.theta = point[2]
        self.theta_real = point[2]
    

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--map_dir', default='../../leaderboard/data/maps', help='folder with created map images')
    argparser.add_argument('--ppm', default=8,  help='pixels per meter, should match the created map')
    argparser.add_argument('--in_path', type=str, help='input xmls or json file')
    argparser.add_argument( '--save_dir', type=str, help='output folder with visualizations')
    args = argparser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    filename = args.in_path.split('/')[-1].split('.')[0]
    extension = args.in_path.split('/')[-1].split('.')[1]
    print(f'Processing {filename}, extension: {extension}')
    if extension == 'xml':
        root = ET.parse(args.in_path).getroot()
        points_ = []

        for type_tag in root.findall('route'):
            tmp_pts_ego = []
            town_ = type_tag.get('town')
            for  wp_ in type_tag.findall('waypoint'):
                sp_x = float(wp_.get('x'))
                sp_y = float(wp_.get('y'))
                tmp_pts_ego.append(Point((sp_x, sp_y, 0)))
            points_.append(tmp_pts_ego)
        
        args.points = points_ 

        map_image = MapImage(carla_map_name=town_, pixels_per_meter=args.ppm, map_dir=args.map_dir)
        map_image.plot_points_nocrop_xml(map_image.viz_surface, args.points, filename + '_' + extension, args.save_dir)
    else:
        with open(args.in_path) as json_file:
            data = json.load(json_file)  
        
        # extract json data
        scenario_list = data["available_scenarios"][0]
        
        for town_, scenarios_list in scenario_list.items():
            args.points = [] 
            map_image = MapImage(carla_map_name=town_, pixels_per_meter=args.ppm, map_dir=args.map_dir)
            for scenario_ in scenarios_list:
                scenario_all_triggers = scenario_["available_event_configurations"]
                tmp_pts_ego = []
                tmp_pts = []

                for scenario_trigger in scenario_all_triggers:
                    scenario_points = {}  
                    scenario_ego_spawn = scenario_trigger['transform']
                    sp_x, sp_y, sp_yaw = float(scenario_ego_spawn['x']), float(scenario_ego_spawn['y']), float(scenario_ego_spawn['yaw'])
                    
                    scenario_points['ego'] = Point((sp_x, sp_y, sp_yaw))
                    tmp_pts.append(scenario_points)                                       
                args.points.append(tmp_pts)
        
            map_image.plot_points_nocrop_json(map_image.viz_surface, args.points, filename + '_' + extension, args.save_dir)
            

if __name__ == '__main__':
    main()
