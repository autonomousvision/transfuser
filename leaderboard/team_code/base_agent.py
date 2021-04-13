import time

import cv2
import carla

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner


# Full scale sensor setup including depth, semantics and lidar
class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self): # extra sensors added by aditya
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_front'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'seg_front'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'depth_front'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'seg_left'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'depth_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_rear'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'seg_rear'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'depth_rear'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'seg_right'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'depth_right'
                    },
                {   
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'id': 'lidar'
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
        self.step += 1

        rgb_front = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        # segmentation
        seg_front = cv2.cvtColor(input_data['seg_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        seg_left = cv2.cvtColor(input_data['seg_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        seg_rear = cv2.cvtColor(input_data['seg_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        seg_right = cv2.cvtColor(input_data['seg_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        
        # depth
        depth_front = cv2.cvtColor(input_data['depth_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_left = cv2.cvtColor(input_data['depth_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_rear = cv2.cvtColor(input_data['depth_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_right = cv2.cvtColor(input_data['depth_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        
        # lidar
        lidar = input_data['lidar'][1]
        

        return {
                'rgb_front': rgb_front,
                'rgb_left': rgb_left,
                'rgb_rear': rgb_rear,
                'rgb_right': rgb_right,
                'seg_front': seg_front,
                'seg_left': seg_left,
                'seg_rear': seg_rear,
                'seg_right': seg_right,
                'depth_front': depth_front,
                'depth_left': depth_left,
                'depth_rear': depth_rear,
                'depth_right': depth_right,
                'lidar': lidar,
                'gps': gps,
                'speed': speed,
                'compass': compass
                }