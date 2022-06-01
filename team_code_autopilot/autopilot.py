import os
import json
import datetime
import pathlib
import time
import statistics
from collections import deque, defaultdict

import math
import numpy as np
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents import autonomous_agent
from new_sensors.nav_planner import PIDController, RoutePlanner, interpolate_trajectory

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'AutoPilot'


class AutoPilot(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        
        self.track = autonomous_agent.Track.MAP
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.save_path = None
        self.render_bev = False
        self.route_index = route_index
        self.frame_rate_sim = 20 # CARLA framerate, used for visualizations

        self.gps_buffer = deque(maxlen=100) # Stores the last x updated gps signals.

        # Dynamics models
        self.frame_rate = 20 # Brame rate used by kinematic bicycle model for forecasting
        self.ego_model     = EgoModel(dt=(1.0 / self.frame_rate))
        self.vehicle_model = EgoModel(dt=(1.0 / self.frame_rate))

        # Configuration
        self.visualize = int(os.environ['DEBUG_CHALLENGE'])
        self.save_freq = self.frame_rate_sim//2 # By default, save once every 10 frames (0.5 seconds)

        # Controllers
        self.steer_buffer_size = 1     # Number of elements to average steering over
        self.target_speed_slow = 3.0	# Speed at junctions, m/s
        self.target_speed_fast = 4.0	# Speed outside junctions, m/s
        self.clip_delta = 0.25			# Max angular error for turn controller
        self.clip_throttle = 0.75		# Max throttle (0-1)
        self.steer_damping = 0.5		# Steer multiplicative reduction while braking
        self.slope_pitch = 10.0			# Pitch above which throttle is increased
        self.slope_throttle = 0.4		# Excess throttle applied on slopes
        self.angle_search_range = 0    # Number of future waypoints to consider in angle search
        self.steer_noise = 1e-3         # Noise added to expert steering angle
        self.steer_buffer = deque(maxlen=self.steer_buffer_size)
        
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._turn_controller_extrapolation = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self._speed_controller_extrapolation = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # Red light detection
        # Coordinates of the center of the red light detector bounding box. In local coordinates of the vehicle, units are meters
        self.center_bb_light_x = -2.0
        self.center_bb_light_y = 0.0
        self.center_bb_light_z = 0.0

        # Extent of the red light detector bounding box. In local coordinates of the vehicle, units are meters. Size are half of the bounding box
        self.extent_bb_light_x = 4.5
        self.extent_bb_light_y = 1.5
        self.extent_bb_light_z = 2.0

        # Obstacle detection
        self.extrapolation_seconds_no_junction = 1.0    # Amount of seconds we look into the future to predict collisions (>= 1 frame)
        self.extrapolation_seconds = 4.0                # Amount of seconds we look into the future to predict collisions at junctions
        self.waypoint_seconds = 4.0                     # Amount of seconds we look into the future to store waypoint labels
        self.detection_radius = 30.0                    # Distance of obstacles (in meters) in which we will check for collisions
        self.light_radius = 15.0                        # Distance of traffic lights considered relevant (in meters)

        # Speed buffer for detecting "stuck" vehicles
        self.vehicle_speed_buffer = defaultdict( lambda: {"velocity": [], "throttle": [], "brake": []})
        self.stuck_buffer_size = 30
        self.stuck_vel_threshold = 0.1
        self.stuck_throttle_threshold = 0.1
        self.stuck_brake_threshold = 0.1

        # Navigation command buffer
        self.commands = deque(maxlen=2)
        self.commands.append(4)
        self.commands.append(4)
        self.far_node_prev = [1e5, 1e5]

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.target_speed = 4.0

        self.angle                = 0.0   # Angle to the next waypoint. Normalized in [-1, 1] corresponding to [-90, 90]
        self.stop_sign_hazard     = False
        self.traffic_light_hazard = False
        self.walker_hazard        = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.vehicle_hazard       = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        self.junction             = False
        self.ignore_stop_signs    = True # Whether to ignore stop signs
        self.cleared_stop_signs = []  # A list of all stop signs that we have cleared
        self.future_states = {}
        
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += f'route{self.route_index}_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print (string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'measurements').mkdir()

    def _init(self, hd_map):
        # Near node
        self.world_map = carla.Map("RouteMap", hd_map[1]['opendrive'])
        trajectory = [item[0].location for item in self._global_plan_world_coord]
        self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)
        
        print("Sparse Waypoints:", len(self._global_plan))
        print("Dense Waypoints:", len(self.dense_route))

        self._waypoint_planner = RoutePlanner(3.5, 50)
        self._waypoint_planner.set_route(self.dense_route, True)
        self._waypoint_planner.save()

        self._waypoint_planner_extrapolation = RoutePlanner(3.5, 50)
        self._waypoint_planner_extrapolation.set_route(self.dense_route, True)
        self._waypoint_planner_extrapolation.save()

        # Far node
        self._command_planner = RoutePlanner(7.5, 50)
        self._command_planner.set_route(self._global_plan, True)

        # Privileged
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        self.initialized = True

    def sensors(self):
        return [{
                    'type': 'sensor.opendrive_map',
                    'reading_frequency': 1e-6,
                    'id': 'hd_map'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def tick(self, input_data):
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        if (math.isnan(compass) == True): # simulation bug
            compass = 0.0

        result = {
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }

        return result

    def run_step(self, input_data, timestamp):
        self.step += 1
        if not self.initialized:
            if ('hd_map' in input_data.keys()):
                self._init(input_data['hd_map'])
            else:
                control = carla.VehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                return control

        control = self._get_control(input_data)
        tick_data_tmp = self.tick(input_data)
        self.update_gps_buffer(control, tick_data_tmp['compass'], tick_data_tmp['speed'])
        return control

    def update_gps_buffer(self, control, theta, speed):
        yaw = np.array([(theta - np.pi/2.0)])
        speed = np.array([speed])
        action = np.array(np.stack([control.steer, control.throttle, control.brake], axis=-1))

        #Update gps locations
        for i in range(len(self.gps_buffer)):
            loc =self.gps_buffer[i]
            loc_temp = np.array([loc[1], -loc[0]]) #Bicycle model uses a different coordinate system
            next_loc_tmp, _, _ = self.ego_model.forward(loc_temp, yaw, speed, action)
            next_loc = np.array([-next_loc_tmp[1], next_loc_tmp[0]])
            self.gps_buffer[i] = next_loc

        return None

    def get_future_states(self):
        return self.future_states

    def _get_control(self, input_data, steer=None, throttle=None,
                        vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):
        # insert missing controls
        if vehicle_hazard is None or light_hazard is None or walker_hazard is None or stop_sign_hazard is None:
            brake = self._get_brake(vehicle_hazard, light_hazard, walker_hazard, stop_sign_hazard) # privileged
        else:
            brake = vehicle_hazard or light_hazard or walker_hazard or stop_sign_hazard

        ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        self.junction = ego_vehicle_waypoint.is_junction

        speed = input_data['speed'][1]['speed']
        target_speed = self.target_speed_slow if self.junction else self.target_speed_fast

        pos = self._get_position(input_data['gps'][1][:2])
        self.gps_buffer.append(pos)
        pos = np.average(self.gps_buffer, axis=0) # Denoised position

        self._waypoint_planner.load()
        waypoint_route = self._waypoint_planner.run_step(pos)
        near_node, near_command = waypoint_route[1] if len(waypoint_route) > 1 else waypoint_route[0] # needs HD map
        self._waypoint_planner.save()
        
        self._waypoint_planner_extrapolation.load()
        self.waypoint_route_extrapolation = self._waypoint_planner_extrapolation.run_step(pos)
        self._waypoint_planner_extrapolation.save()

        if throttle is None:
            throttle = self._get_throttle(brake, target_speed, speed)

            # hack for steep slopes
            if (self._vehicle.get_transform().rotation.pitch > self.slope_pitch):
                throttle += self.slope_throttle

        if steer is None:
            theta = input_data['imu'][1][-1]
            steer = self._get_steer(brake, waypoint_route, pos, theta, speed)
            steer_extrapolation = self._get_steer_extrapolation(waypoint_route, pos, theta, speed)

        self.steer_buffer.append(steer)

        control = carla.VehicleControl()
        control.steer = np.mean(self.steer_buffer) + self.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        self.target_speed = target_speed

        self._save_waypoints()

        if((self.step % self.save_freq == 0) and (self.save_path is not None)):
            command_route = self._command_planner.run_step(pos)
            # far command is not always accurate
            far_node, far_command  = command_route[1] if len(command_route) > 1 else command_route[0]
            if (far_node != self.far_node_prev).all():
                self.far_node_prev = far_node
                self.commands.append(far_command.value)

            if self.render_bev==False:
                tick_data = self.tick(input_data)
            else:
                tick_data = self.tick(input_data, self.future_states)
            self.save(far_node, steer, throttle, brake, target_speed, tick_data)

        return control

    def save(self, far_node, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // self.save_freq

        pos = self._get_position(tick_data['gps'])
        theta = tick_data['compass']
        speed = tick_data['speed']

        waypoints = []
        for i, box in enumerate(self.future_states['ego']):
            if (i+1)%(self.frame_rate/2)==0:
                box_x = -box.location.y
                box_y = box.location.x
                box_theta = box.rotation.yaw*np.pi/180.0 + np.pi/2
                if box_theta < 0:
                    box_theta += 2*np.pi
                waypoints.append((box_x, box_y, box_theta))

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': target_speed,
                'x_command': far_node[0],
                'y_command': far_node[1],
                'command': self.commands[-2],
                'waypoints': waypoints,
                'steer': steer,
                'throttle': throttle,
                'brake': brake,
                'junction':         self.junction,
                'vehicle_hazard':   self.vehicle_hazard,
                'light_hazard':     self.traffic_light_hazard,
                'walker_hazard':    self.walker_hazard,
                'stop_sign_hazard': self.stop_sign_hazard,
                'angle':            self.angle,
                'ego_matrix': self._vehicle.get_transform().get_matrix()
                }

        measurements_file = self.save_path / 'measurements' / ('%04d.json' % frame)
        with open(measurements_file, 'w') as f:
            json.dump(data, f, indent=4)

    def destroy(self):
        pass

    def _get_steer(self, brake, route, pos, theta, speed, restore=True):
        if self._waypoint_planner.is_last: # end of route
            angle = 0.0

        if((speed < 0.01) and (brake == True)): # prevent accumulation
            angle = 0.0

        if len(route) == 1:
            target = route[0][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        elif self.angle_search_range <= 2:
            target = route[1][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        else:
            search_range = min([len(route), self.angle_search_range])
            for i in range(1, search_range):
                target = route[i][0]
                angle_unnorm = self._get_angle_to(pos, theta, target)
                angle_new = angle_unnorm / 90
                if i==1:
                    angle = angle_new
                if np.abs(angle_new) < np.abs(angle):
                    angle = angle_new

        self.angle = angle

        if restore: self._turn_controller.load()
        steer = self._turn_controller.step(angle)
        if restore: self._turn_controller.save()

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        if brake:
            steer *= self.steer_damping

        return steer

    def _get_steer_extrapolation(self, route, pos, theta, speed, restore=True):
        if self._waypoint_planner_extrapolation.is_last: # end of route
            angle = 0.0

        if len(route) == 1:
            target = route[0][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        elif self.angle_search_range <= 2:
            target = route[1][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        else:
            search_range = min([len(route), self.angle_search_range])
            for i in range(1, search_range):
                target = route[i][0]
                angle_unnorm = self._get_angle_to(pos, theta, target)
                angle_new = angle_unnorm / 90
                if i==1:
                    angle = angle_new
                if np.abs(angle_new) < np.abs(angle):
                    angle = angle_new

        self.angle = angle

        if restore: self._turn_controller_extrapolation.load()
        steer = self._turn_controller_extrapolation.step(angle)
        if restore: self._turn_controller_extrapolation.save()

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer

    def _get_throttle(self, brake, target_speed, speed, restore=True):
        target_speed = target_speed if not brake else 0.0

        if self._waypoint_planner.is_last: # end of route
            target_speed = 0.0

        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)

        if restore: self._speed_controller.load()
        throttle = self._speed_controller.step(delta)
        if restore: self._speed_controller.save()

        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        if brake:
            throttle = 0.0

        return throttle

    def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
        if self._waypoint_planner_extrapolation.is_last: # end of route
            target_speed = 0.0

        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)

        if restore: self._speed_controller_extrapolation.load()
        throttle = self._speed_controller_extrapolation.step(delta)
        if restore: self._speed_controller_extrapolation.save()

        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        return throttle

    def _get_brake(self, vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):
        actors = self._world.get_actors()
        speed = self._get_forward_speed()
        
        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()

        vehicles = actors.filter('*vehicle*')

        # -----------------------------------------------------------
        # Red light detection
        # -----------------------------------------------------------
        if light_hazard is None:
            light_hazard = False
            self._active_traffic_light = None
            _traffic_lights = self.get_nearby_object(vehicle_location, actors.filter('*traffic_light*'), self.light_radius)
            
            center_light_detector_bb = vehicle_transform.transform(carla.Location(x=self.center_bb_light_x, y=self.center_bb_light_y, z=self.center_bb_light_z))
            extent_light_detector_bb = carla.Vector3D(x=self.extent_bb_light_x, y=self.extent_bb_light_y, z=self.extent_bb_light_z)
            light_detector_bb = carla.BoundingBox(center_light_detector_bb, extent_light_detector_bb)
            light_detector_bb.rotation = vehicle_transform.rotation
            color2 = carla.Color(255, 255, 255, 255)
            for light in _traffic_lights:
                if   (light.state == carla.libcarla.TrafficLightState.Red):
                    color = carla.Color(255, 0, 0, 255)
                elif (light.state == carla.libcarla.TrafficLightState.Yellow):
                    color = carla.Color(255, 255, 0, 255)
                elif (light.state == carla.libcarla.TrafficLightState.Green):
                    color = carla.Color(0, 255, 0, 255)
                elif (light.state == carla.libcarla.TrafficLightState.Off):
                    color = carla.Color(0, 0, 0, 255)
                else: # unknown
                    color = carla.Color(0, 0, 255, 255)

                size = 0.1 # size of the points and bounding boxes used for visualization
                # box in which we will look for traffic light triggers.            
                center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
                center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
                length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y, light.trigger_volume.extent.z)
                transform = carla.Transform(center_bounding_box) # can only create a bounding box from a transform.location, not from a location
                bounding_box = carla.BoundingBox(transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch = light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                                    yaw   = light.trigger_volume.rotation.yaw   + gloabl_rot.yaw,
                                                    roll  = light.trigger_volume.rotation.roll  + gloabl_rot.roll)
                if (self.visualize == 1):
                    self._world.debug.draw_point(vehicle_location + carla.Location(z=2.0), size=size, color=color, life_time=(1.0/self.frame_rate_sim))
                    self._world.debug.draw_point(center_bounding_box + carla.Location(z=2.0), size=size, color=color, life_time=(1.0/self.frame_rate_sim))
                    self._world.debug.draw_box(box=bounding_box, rotation= bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0/self.frame_rate_sim))

                if(self.check_obb_intersection(light_detector_bb, bounding_box) == True):
                    if ((light.state == carla.libcarla.TrafficLightState.Red)
                        or (light.state == carla.libcarla.TrafficLightState.Yellow)):
                        self._active_traffic_light = light
                        light_hazard = True
                        color2 = carla.Color(255, 0, 0, 255)

            if (self.visualize == 1):
                self._world.debug.draw_box(box=light_detector_bb, rotation=light_detector_bb.rotation, thickness=0.1, color=color2, life_time=(1.0 / self.frame_rate_sim))

        #-----------------------------------------------------------
        # Stop sign detection
        #-----------------------------------------------------------
        if stop_sign_hazard is None:
            stop_sign_hazard = False
            if not self.ignore_stop_signs:
                stop_signs     = self.get_nearby_object(vehicle_location, actors.filter('*stop*'), self.light_radius)
                center_vehicle_stop_sign_detector_bb   = vehicle_transform.transform(self._vehicle.bounding_box.location)
                extent_vehicle_stop_sign_detector_bb   = self._vehicle.bounding_box.extent
                vehicle_stop_sign_detector_bb          = carla.BoundingBox(center_vehicle_stop_sign_detector_bb, extent_vehicle_stop_sign_detector_bb)
                vehicle_stop_sign_detector_bb.rotation = vehicle_transform.rotation
                
                for stop_sign in stop_signs:
                    center_bb_stop_sign    = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)            
                    length_bb_stop_sign    = carla.Vector3D(stop_sign.trigger_volume.extent.x, stop_sign.trigger_volume.extent.y, stop_sign.trigger_volume.extent.z)
                    transform_stop_sign    = carla.Transform(center_bb_stop_sign)
                    bounding_box_stop_sign = carla.BoundingBox(transform_stop_sign.location, length_bb_stop_sign)
                    rotation_stop_sign     = stop_sign.get_transform().rotation
                    bounding_box_stop_sign.rotation = carla.Rotation(pitch=stop_sign.trigger_volume.rotation.pitch + rotation_stop_sign.pitch, 
                                                                    yaw  =stop_sign.trigger_volume.rotation.yaw   + rotation_stop_sign.yaw,
                                                                    roll =stop_sign.trigger_volume.rotation.roll  + rotation_stop_sign.roll)

                    color = carla.Color(0, 255, 0, 255)

                    if (self.check_obb_intersection(vehicle_stop_sign_detector_bb, bounding_box_stop_sign) == True):
                        if(not (stop_sign.id in self.cleared_stop_signs)):
                            if((speed * 3.6) > 0.0): #Conversion from m/s to km/h
                                stop_sign_hazard = True
                                color = carla.Color(255, 0, 0, 255)
                            else:
                                self.cleared_stop_signs.append(stop_sign.id)
                    if (self.visualize == 1):
                        self._world.debug.draw_point(center_bb_stop_sign + carla.Location(z=2.0), size=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                        self._world.debug.draw_box(box=bounding_box_stop_sign, rotation=bounding_box_stop_sign.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                
                # reset past cleared stop signs
                for cleared_stop_sign in self.cleared_stop_signs:
                    remove_stop_sign = True
                    for stop_sign in stop_signs:
                        if(stop_sign.id == cleared_stop_sign):
                            remove_stop_sign = False # stop sign is still around us hence it might be active
                    if(remove_stop_sign == True):
                        self.cleared_stop_signs.remove(cleared_stop_sign)

        # -----------------------------------------------------------
        # Obstacle detection
        # -----------------------------------------------------------
        if vehicle_hazard is None or walker_hazard is None:
            vehicle_hazard = False
            self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
            extrapolation_seconds   = self.extrapolation_seconds  # amount of seconds we look into the future to predict collisions
            detection_radius        = self.detection_radius       # distance in which we check for collisions
            number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
            number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate)
            color = carla.Color(0, 255, 0, 255)

            # -----------------------------------------------------------
            # Walker detection
            # -----------------------------------------------------------
            walkers = actors.filter('*walker*')
            walker_hazard  = False
            self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
            nearby_walkers = []
            for walker in walkers:
                if (walker.get_location().distance(vehicle_location) < detection_radius):
                    walker_future_bbs = []
                    walker_transform = walker.get_transform()
                    walker_velocity = walker.get_velocity()
                    walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)  # In m/s
                    walker_location = walker_transform.location
                    walker_direction = walker.get_control().direction

                    for i in range(number_of_future_frames):
                        if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break

                        # NOTE for perf. optimization: Could also just add velocity.x instead might be a bit faster
                        new_x = walker_location.x + walker_direction.x * walker_speed * (1.0 / self.frame_rate)
                        new_y = walker_location.y + walker_direction.y * walker_speed * (1.0 / self.frame_rate)
                        new_z = walker_location.z + walker_direction.z * walker_speed * (1.0 / self.frame_rate)
                        walker_location = carla.Location(new_x, new_y, new_z)
                        if (self.visualize == 1):
                            self._world.debug.draw_point(walker_location, size=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))

                        transform = carla.Transform(walker_location)
                        bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(pitch = walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
                                                            yaw   = walker.bounding_box.rotation.yaw   + walker_transform.rotation.yaw,
                                                            roll  = walker.bounding_box.rotation.roll  + walker_transform.rotation.roll)

                        color = carla.Color(0, 0, 255, 255)
                        if (self.visualize == 1):
                            self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                        walker_future_bbs.append(bounding_box)
                    nearby_walkers.append(walker_future_bbs)

            # -----------------------------------------------------------
            # Vehicle detection
            # -----------------------------------------------------------
            nearby_vehicles = {}
            tmp_near_vehicle_id = []
            tmp_stucked_vehicle_id = []
            for vehicle in vehicles:
                if (vehicle.id == self._vehicle.id):
                    continue
                if (vehicle.get_location().distance(vehicle_location) < detection_radius):
                    tmp_near_vehicle_id.append(vehicle.id)
                    veh_future_bbs    = []
                    traffic_transform = vehicle.get_transform()
                    traffic_control   = vehicle.get_control()
                    traffic_velocity  = vehicle.get_velocity()
                    traffic_speed     = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity) # In m/s

                    self.vehicle_speed_buffer[vehicle.id]["velocity"].append(traffic_speed)
                    self.vehicle_speed_buffer[vehicle.id]["throttle"].append(traffic_control.throttle)
                    self.vehicle_speed_buffer[vehicle.id]["brake"].append(traffic_control.brake)
                    if len(self.vehicle_speed_buffer[vehicle.id]["velocity"]) > self.stuck_buffer_size:
                        self.vehicle_speed_buffer[vehicle.id]["velocity"] = self.vehicle_speed_buffer[vehicle.id]["velocity"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]["throttle"] = self.vehicle_speed_buffer[vehicle.id]["throttle"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]["brake"] = self.vehicle_speed_buffer[vehicle.id]["brake"][-self.stuck_buffer_size:]


                    next_loc   = np.array([traffic_transform.location.x, traffic_transform.location.y])
                    action     = np.array(np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
                    next_yaw   = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                    next_speed = np.array([traffic_speed])
                    
                    for i in range(number_of_future_frames):
                        if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break

                        next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)
                        if (self.visualize == 1):
                            self._world.debug.draw_point(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z) + carla.Location(z=3.0), size=0.1, 
                                                        color=color, life_time=(1.0 / self.frame_rate_sim))

                        delta_yaws = next_yaw.item() * 180.0 / np.pi

                        transform             = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                        bounding_box          = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch),
                                                            yaw=float(delta_yaws),
                                                            roll=float(traffic_transform.rotation.roll))

                        color = carla.Color(0, 0, 255, 255)
                        if (self.visualize == 1):
                            self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                        veh_future_bbs.append(bounding_box)
                    
                    if (statistics.mean(self.vehicle_speed_buffer[vehicle.id]["velocity"]) < self.stuck_vel_threshold
                            and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["throttle"]) > self.stuck_throttle_threshold
                            and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["brake"]) < self.stuck_brake_threshold):
                        tmp_stucked_vehicle_id.append(vehicle.id)

                    nearby_vehicles[vehicle.id] = veh_future_bbs

            # delete old vehicles
            to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
            for d in to_delete:
                del self.vehicle_speed_buffer[d]
            if (self.visualize == 1):
                self._world.debug.draw_point(vehicle_location + carla.Location(z=3.0), size=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))

            # -----------------------------------------------------------
            # Intersection checks with ego vehicle
            # -----------------------------------------------------------

            next_loc_no_brake   = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
            next_yaw_no_brake   = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
            next_speed_no_brake = np.array([speed])

            #NOTE intentionally set ego vehicle to move at the target speed (we want to know if there is an intersection if we would not brake)
            throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)
            action_no_brake     = np.array(np.stack([self.steer, throttle_extrapolation, 0.0], axis=-1))

            back_only_vehicle_id = []
            ego_future = []

            for i in range(number_of_future_frames):
                if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                    alpha = 255
                    color_value = 50
                    break
                else:
                    alpha = 50
                    color_value = 255

                # calculate ego vehicle bounding box for the next timestep. We don't consider timestep 0 because it is from the past and has already happened.
                next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake = self.ego_model.forward(next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake, action_no_brake)
                next_loc_no_brake_temp = np.array([-next_loc_no_brake[1], next_loc_no_brake[0]])
                next_yaw_no_brake_temp = next_yaw_no_brake.item() + np.pi/2 # in global coordinates

                waypoint_route_extrapolation_temp = self._waypoint_planner_extrapolation.run_step(next_loc_no_brake_temp)
                steer_extrapolation_temp = self._get_steer_extrapolation(waypoint_route_extrapolation_temp, next_loc_no_brake_temp, next_yaw_no_brake_temp, next_speed_no_brake, restore=False)
                throttle_extrapolation_temp = self._get_throttle_extrapolation(self.target_speed, next_speed_no_brake, restore=False)
                brake_extrapolation_temp = 1.0 if self._waypoint_planner_extrapolation.is_last else 0.0
                action_no_brake = np.array(np.stack([steer_extrapolation_temp, float(throttle_extrapolation_temp), brake_extrapolation_temp], axis=-1))

                if (self.visualize == 1):
                    if self.junction or i <= number_of_future_frames_no_junction:
                        self._world.debug.draw_point(carla.Location(x=next_loc_no_brake[0].item(), y=next_loc_no_brake[1].item(), z=vehicle_transform.location.z) + carla.Location(z=3.0),size=0.1, color=color, life_time=(1.0 / self.frame_rate))

                delta_yaws_no_brake = next_yaw_no_brake.item() * 180.0 / np.pi
                cosine = np.cos(next_yaw_no_brake.item())
                sine = np.sin(next_yaw_no_brake.item())

                extent           = self._vehicle.bounding_box.extent
                extent_org       = self._vehicle.bounding_box.extent
                extent.x         = extent.x / 2.

                # front half
                transform             = carla.Transform(carla.Location(x=next_loc_no_brake[0].item()+extent.x*cosine, y=next_loc_no_brake[1].item()+extent.y*sine, z=vehicle_transform.location.z))
                bounding_box          = carla.BoundingBox(transform.location, extent)
                bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws_no_brake), roll=float(vehicle_transform.rotation.roll))

                # back half
                transform_back             = carla.Transform(carla.Location(x=next_loc_no_brake[0].item()-extent.x*cosine, y=next_loc_no_brake[1].item()-extent.y*sine, z=vehicle_transform.location.z))
                bounding_box_back          = carla.BoundingBox(transform_back.location, extent)
                bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws_no_brake), roll=float(vehicle_transform.rotation.roll))
                
                color = carla.Color(0, color_value, 0, alpha)
                color2 = carla.Color(0, color_value, color_value, alpha)

                i_stuck = i
                for id, traffic_participant in nearby_vehicles.items():
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break
                    if id in tmp_stucked_vehicle_id:
                        i_stuck = 0
                    back_intersect = (self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) == True)
                    front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True)
                    if id in back_only_vehicle_id:
                        back_only_vehicle_id.remove(id)
                        if back_intersect:
                            back_only_vehicle_id.append(id)
                        continue
                    if back_intersect and not front_intersect:
                        back_only_vehicle_id.append(id)
                    if front_intersect:
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction==True or i <= number_of_future_frames_no_junction:
                            vehicle_hazard = True
                        self.vehicle_hazard[i] = True
                    
                for walker in nearby_walkers:
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break
                    if (self.check_obb_intersection(bounding_box, walker[i]) == True):
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction==True or i <= number_of_future_frames_no_junction:
                            walker_hazard = True
                        self.walker_hazard[i] = True

                if (self.visualize == 1):
                    self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                    self._world.debug.draw_box(box=bounding_box_back, rotation=bounding_box.rotation, thickness=0.1, color=color2, life_time=(1.0 / self.frame_rate_sim))

            # add safety bounding box in front. If there is anything in there we won't start driving
            color = carla.Color(0, 255, 0, 255)
            
            bremsweg = ((speed * 3.6) / 10.0)**2 / 2.0 # Bremsweg formula for emergency break
            safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0) # plus one meter is the car.
            
            center_safety_box     = vehicle_transform.transform(carla.Location(x=safety_x, y=0.0, z=0.0))
            bounding_box          = carla.BoundingBox(center_safety_box, self._vehicle.bounding_box.extent)
            bounding_box.rotation = vehicle_transform.rotation
            
            for _, traffic_participant in nearby_vehicles.items():
                if (self.check_obb_intersection(bounding_box, traffic_participant[0]) == True): # check the first BB of the traffic participant. We don't extrapolate into the future here.
                    color = carla.Color(255, 0, 0, 255)
                    vehicle_hazard = True
                    self.vehicle_hazard[0] = True

            for walker in nearby_walkers:
                if (self.check_obb_intersection(bounding_box, walker[0]) == True): # check the first BB of the traffic participant. We don't extrapolate into the future here.
                    color = carla.Color(255, 0, 0, 255)
                    walker_hazard = True
                    self.walker_hazard[0] = True

            if (self.visualize == 1):
                self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))

            self.future_states = {'walker': nearby_walkers, 'vehicle': nearby_vehicles}

        else:
            self.vehicle_hazard = vehicle_hazard
            self.walker_hazard = walker_hazard

        self.stop_sign_hazard     = stop_sign_hazard
        self.traffic_light_hazard = light_hazard
        
        return (vehicle_hazard or light_hazard or walker_hazard or stop_sign_hazard)

    def _intersection_check(self, ego_wps):
        actors = self._world.get_actors()
        speed = self._get_forward_speed()
        
        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()

        vehicles = actors.filter('*vehicle*')

       
        # -----------------------------------------------------------
        # Obstacle detection
        # -----------------------------------------------------------
        vehicle_hazard = False
        self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        extrapolation_seconds   = self.extrapolation_seconds  # amount of seconds we look into the future to predict collisions
        detection_radius        = self.detection_radius       # distance in which we check for collisions
        number_of_future_frames = int(extrapolation_seconds * self.frame_rate)
        number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate)
        color = carla.Color(0, 255, 0, 255)

        # -----------------------------------------------------------
        # Walker detection
        # -----------------------------------------------------------
        walkers = actors.filter('*walker*')
        walker_hazard  = False
        self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate))]
        nearby_walkers = []
        for walker in walkers:
            if (walker.get_location().distance(vehicle_location) < detection_radius):
                walker_future_bbs = []
                walker_transform = walker.get_transform()
                walker_velocity = walker.get_velocity()
                walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)  # In m/s
                walker_location = walker_transform.location
                walker_direction = walker.get_control().direction

                for i in range(number_of_future_frames):
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                        break

                    # NOTE for perf. optimization: Could also just add velocity.x instead might be a bit faster
                    new_x = walker_location.x + walker_direction.x * walker_speed * (1.0 / self.frame_rate)
                    new_y = walker_location.y + walker_direction.y * walker_speed * (1.0 / self.frame_rate)
                    new_z = walker_location.z + walker_direction.z * walker_speed * (1.0 / self.frame_rate)
                    walker_location = carla.Location(new_x, new_y, new_z)
                    if (self.visualize == 1):
                        self._world.debug.draw_point(walker_location, size=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))

                    transform = carla.Transform(walker_location)
                    bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
                    bounding_box.rotation = carla.Rotation(pitch = walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
                                                        yaw   = walker.bounding_box.rotation.yaw   + walker_transform.rotation.yaw,
                                                        roll  = walker.bounding_box.rotation.roll  + walker_transform.rotation.roll)

                    color = carla.Color(0, 0, 255, 255)
                    if (self.visualize == 1):
                        self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                    walker_future_bbs.append(bounding_box)
                nearby_walkers.append(walker_future_bbs)

        # -----------------------------------------------------------
        # Vehicle detection
        # -----------------------------------------------------------
        nearby_vehicles = {}
        tmp_near_vehicle_id = []
        tmp_stucked_vehicle_id = []
        for vehicle in vehicles:
            if (vehicle.id == self._vehicle.id):
                continue
            if (vehicle.get_location().distance(vehicle_location) < detection_radius):
                tmp_near_vehicle_id.append(vehicle.id)
                veh_future_bbs    = []
                traffic_transform = vehicle.get_transform()
                traffic_control   = vehicle.get_control()
                traffic_velocity  = vehicle.get_velocity()
                traffic_speed     = self._get_forward_speed(transform=traffic_transform, velocity=traffic_velocity) # In m/s

                self.vehicle_speed_buffer[vehicle.id]["velocity"].append(traffic_speed)
                self.vehicle_speed_buffer[vehicle.id]["throttle"].append(traffic_control.throttle)
                self.vehicle_speed_buffer[vehicle.id]["brake"].append(traffic_control.brake)
                if len(self.vehicle_speed_buffer[vehicle.id]["velocity"]) > self.stuck_buffer_size:
                    self.vehicle_speed_buffer[vehicle.id]["velocity"] = self.vehicle_speed_buffer[vehicle.id]["velocity"][-self.stuck_buffer_size:]
                    self.vehicle_speed_buffer[vehicle.id]["throttle"] = self.vehicle_speed_buffer[vehicle.id]["throttle"][-self.stuck_buffer_size:]
                    self.vehicle_speed_buffer[vehicle.id]["brake"] = self.vehicle_speed_buffer[vehicle.id]["brake"][-self.stuck_buffer_size:]


                next_loc   = np.array([traffic_transform.location.x, traffic_transform.location.y])
                action     = np.array(np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
                next_yaw   = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                next_speed = np.array([traffic_speed])
                
                for i in range(number_of_future_frames):
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                        break

                    next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed, action)
                    if (self.visualize == 1):
                        self._world.debug.draw_point(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z) + carla.Location(z=3.0), size=0.1, 
                                                    color=color, life_time=(1.0 / self.frame_rate_sim))

                    delta_yaws = next_yaw.item() * 180.0 / np.pi

                    transform             = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                    bounding_box          = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                    bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch),
                                                        yaw=float(delta_yaws),
                                                        roll=float(traffic_transform.rotation.roll))

                    color = carla.Color(0, 0, 255, 255)
                    if (self.visualize == 1):
                        self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                    veh_future_bbs.append(bounding_box)
                
                if (statistics.mean(self.vehicle_speed_buffer[vehicle.id]["velocity"]) < self.stuck_vel_threshold
                        and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["throttle"]) > self.stuck_throttle_threshold
                        and statistics.mean(self.vehicle_speed_buffer[vehicle.id]["brake"]) < self.stuck_brake_threshold):
                    tmp_stucked_vehicle_id.append(vehicle.id)

                nearby_vehicles[vehicle.id] = veh_future_bbs

        # delete old vehicles
        to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
        for d in to_delete:
            del self.vehicle_speed_buffer[d]
        if (self.visualize == 1):
            self._world.debug.draw_point(vehicle_location + carla.Location(z=3.0), size=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))

        # -----------------------------------------------------------
        # Intersection checks with ego vehicle
        # -----------------------------------------------------------
        number_of_interpolation_frames = self.frame_rate // 2 #TODO only checked for 20fps
        cur_loc   = np.array([[vehicle_transform.location.x, vehicle_transform.location.y]])
        cur_loc_ego   = np.array([[0, 0]])

        vehicl_yaw = (vehicle_transform.rotation.yaw + 90)* np.pi / 180
        rotation = np.array(
                [[np.cos(vehicl_yaw), np.sin(vehicl_yaw)],
                [-np.sin(vehicl_yaw),  np.cos(vehicl_yaw)]]
            )
        # tp = target_point[0] @ rotation

        future_loc = cur_loc + ego_wps[0]@ rotation
        all_locs = np.append(cur_loc, future_loc, axis=0)
        all_locs_ego = np.append(cur_loc_ego, ego_wps[0], axis=0)
        cur_yaw   = np.array([(vehicle_transform.rotation.yaw) / 180.0 * np.pi])
        prev_yaw = cur_yaw
        # next_speed_no_brake = np.array([speed])

        #NOTE intentionally set ego vehicle to move at the target speed (we want to know if there is an intersection if we would not brake)
        # throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)
        # action_no_brake     = np.array(np.stack([self.steer, throttle_extrapolation, 0.0], axis=-1))

        back_only_vehicle_id = []
        # ego_future = []

        for i in range(1, 1+ego_wps.shape[1]): # TODO: check dimension!!!
            if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                alpha = 255
                color_value = 50
                break
            else:
                alpha = 50
                color_value = 255

            delta_yaw = math.atan2(all_locs_ego[i][0]-all_locs_ego[i-1][0], all_locs_ego[i][1]-all_locs_ego[i-1][1])
            next_yaw = cur_yaw - delta_yaw

            for k in range(number_of_interpolation_frames):

                tmp_loc = all_locs[i-1] + (all_locs[i]-all_locs[i-1])/number_of_interpolation_frames * k
                tmp_yaw = prev_yaw + (next_yaw-prev_yaw)/number_of_interpolation_frames * k
                # cur_yaw = next_yaw # + delta_yaw
                # next_yaw = np.array([delta_yaw]) #+ delta_yaw

                if (self.visualize == 1):
                    if self.junction or i <= number_of_future_frames_no_junction:
                        self._world.debug.draw_point(carla.Location(x=tmp_loc[0].item(), y=tmp_loc[1].item(), z=vehicle_transform.location.z) + carla.Location(z=3.0),size=0.1, color=color, life_time=(1.0 / self.frame_rate))

                next_yaw_deg = tmp_yaw.item() * 180.0 / np.pi
                cosine = np.cos(tmp_yaw.item())
                sine = np.sin(tmp_yaw.item())

                extent           = self._vehicle.bounding_box.extent
                extent.x         = extent.x / 2.

                # front half
                transform             = carla.Transform(carla.Location(x=tmp_loc[0].item()+extent.x*cosine, y=tmp_loc[1].item()+extent.y*sine, z=vehicle_transform.location.z))
                bounding_box          = carla.BoundingBox(transform.location, extent)
                bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(next_yaw_deg), roll=float(vehicle_transform.rotation.roll))

                # back half
                transform_back             = carla.Transform(carla.Location(x=tmp_loc[0].item()-extent.x*cosine, y=tmp_loc[1].item()-extent.y*sine, z=vehicle_transform.location.z))
                bounding_box_back          = carla.BoundingBox(transform_back.location, extent)
                bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(next_yaw_deg), roll=float(vehicle_transform.rotation.roll))
                
                color = carla.Color(0, color_value, 0, alpha)
                color2 = carla.Color(0, color_value, color_value, alpha)

                index = k + (i-1) * number_of_interpolation_frames
                i_stuck = index

                for id, traffic_participant in nearby_vehicles.items():
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break
                    if id in tmp_stucked_vehicle_id:
                        i_stuck = 0
                    back_intersect = (self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) == True)
                    front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True)
                    if id in back_only_vehicle_id:
                        back_only_vehicle_id.remove(id)
                        if back_intersect:
                            back_only_vehicle_id.append(id)
                        continue
                    if back_intersect and not front_intersect:
                        back_only_vehicle_id.append(id)
                    if front_intersect:
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction==True or i <= number_of_future_frames_no_junction:
                            vehicle_hazard = True
                        self.vehicle_hazard[i] = True
                    
                for walker in nearby_walkers:
                    if self.render_bev==False and self.junction==False and i > number_of_future_frames_no_junction:
                            break
                    if (self.check_obb_intersection(bounding_box, walker[i]) == True):
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction==True or i <= number_of_future_frames_no_junction:
                            walker_hazard = True
                        self.walker_hazard[i] = True

                if (self.visualize == 1):
                    self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))
                    self._world.debug.draw_box(box=bounding_box_back, rotation=bounding_box.rotation, thickness=0.1, color=color2, life_time=(1.0 / self.frame_rate_sim))

                prev_yaw = next_yaw

        if (self.visualize == 1):
            self._world.debug.draw_box(box=bounding_box, rotation=bounding_box.rotation, thickness=0.1, color=color, life_time=(1.0 / self.frame_rate_sim))

        return (vehicle_hazard or walker_hazard)

    def _save_waypoints(self):
        speed = self._get_forward_speed()
        number_of_future_frames = int(self.waypoint_seconds * self.frame_rate)
        vehicle_transform = self._vehicle.get_transform()
        next_loc   = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
        next_yaw   = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
        next_speed = np.array([speed])
        action     = np.array(np.stack([self.steer, self.throttle, self.brake], axis=-1))
        ego_future = []

        for i in range(number_of_future_frames):
            # calculate ego vehicle bounding box for the next timestep. We don't consider timestep 0 because it is from the past and has already happened.
            next_loc, next_yaw, next_speed = self.ego_model.forward(next_loc, next_yaw, next_speed, action)
            next_loc_temp = np.array([-next_loc[1], next_loc[0]])
            next_yaw_temp = next_yaw.item() + np.pi/2 # in global coordinates

            waypoint_route_temp = self._waypoint_planner.run_step(next_loc_temp)
            steer_temp = self._get_steer(self.brake, waypoint_route_temp, next_loc_temp, next_yaw_temp, next_speed, restore=False)
            throttle_temp = self._get_throttle(self.brake, self.target_speed, next_speed, restore=False)
            brake_temp = 1.0 if self._waypoint_planner.is_last else self.brake
            action = np.array(np.stack([steer_temp, float(throttle_temp), brake_temp], axis=-1))
            
            delta_yaws = next_yaw.item() * 180.0 / np.pi
            extent           = self._vehicle.bounding_box.extent
            extent_org       = self._vehicle.bounding_box.extent
            extent.x         = extent.x / 2.

            # whole and without 
            transform_whole            = carla.Transform(carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=vehicle_transform.location.z))
            bounding_box_whole           = carla.BoundingBox(transform_whole.location, extent_org)
            bounding_box_whole.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch), yaw=float(delta_yaws), roll=float(vehicle_transform.rotation.roll))
            ego_future.append(bounding_box_whole)

        self.future_states['ego'] = ego_future

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _get_position(self, gps):
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps

    def dot_product(self, vector1, vector2):
        return (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z)

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y, y=vector1.z * vector2.x - vector1.x * vector2.z, z=vector1.x * vector2.y - vector1.y * vector2.x)

    def get_separating_plane(self, rPos, plane, obb1, obb2):
        ''' Checks if there is a seperating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        '''
        return (abs(self.dot_product(rPos, plane)) > (abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_right_vector()   * obb1.extent.y), plane)) +
                                                      abs(self.dot_product((obb1.rotation.get_up_vector()      * obb1.extent.z), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_right_vector()   * obb2.extent.y), plane)) +
                                                      abs(self.dot_product((obb2.rotation.get_up_vector()      * obb2.extent.z), plane)))
                )
    
    def check_obb_intersection(self, obb1, obb2):
        RPos = obb2.location - obb1.location
        return not(self.get_separating_plane(RPos, obb1.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb1.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_forward_vector(), obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_right_vector(),   obb1, obb2) or
                   self.get_separating_plane(RPos, obb2.rotation.get_up_vector(),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector()  , obb2.rotation.get_up_vector()),      obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_forward_vector()), obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_right_vector()),   obb1, obb2) or
                   self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector()     , obb2.rotation.get_up_vector()),      obb1, obb2))

    # Readable version of get angle to. Don't use.
    def _get_angle_to_slow(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle
        return angle

    # Optimized version of get _get_angle_to. A lot faster, since calculations are done in math and not numpy
    def _get_angle_to(self, pos, theta, target): # 2 - 3 mu
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        diff = target - pos
        aim_0 = ( cos_theta * diff[0] + sin_theta * diff[1])
        aim_1 = (-sin_theta * diff[0] + cos_theta * diff[1])

        angle = -math.degrees(math.atan2(-aim_1, aim_0))
        angle = np.float_(angle) # So that the optimized function has the same datatype as output.
        return angle

    def get_nearby_object(self, vehicle_position, actor_list, radius):
        nearby_objects = []
        for actor in actor_list:
            trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) < radius):
                nearby_objects.append(actor)
        return nearby_objects


class EgoModel():
    def __init__(self, dt=1. / 4):
        self.dt = dt

        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb = -0.090769015
        self.rear_wb = 1.4178275

        self.steer_gain = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs, yaws, spds, acts):
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds
