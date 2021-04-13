#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import time
from threading import Thread
import cv2
import numpy as np

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla

from srunner.autoagents.autonomous_agent import AutonomousAgent


class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, parent):
        self.quit = False
        self._parent = parent
        self._width = 800
        self._height = 600
        self._throttle_delta = 0.05
        self._steering_delta = 0.01
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run(self):
        """
        Run the GUI
        """
        while not self._parent.agent_engaged and not self.quit:
            time.sleep(0.5)

        controller = KeyboardControl()
        while not self.quit:
            self._clock.tick_busy_loop(20)
            controller.parse_events(self._parent.current_control, self._clock)
            # Process events
            pygame.event.pump()

            # process sensor data
            input_data = self._parent.sensor_interface.get_data()
            image_center = input_data['Center'][1][:, :, -2::-1]
            image_left = input_data['Left'][1][:, :, -2::-1]
            image_right = input_data['Right'][1][:, :, -2::-1]
            image_rear = input_data['Rear'][1][:, :, -2::-1]

            top_row = np.hstack((image_left, image_center, image_right))
            bottom_row = np.hstack((0 * image_rear, image_rear, 0 * image_rear))
            comp_image = np.vstack((top_row, bottom_row))
            # resize image
            image_rescaled = cv2.resize(comp_image, dsize=(self._width, self._height), interpolation=cv2.INTER_CUBIC)

            # display image
            self._surface = pygame.surfarray.make_surface(image_rescaled.swapaxes(0, 1))
            if self._surface is not None:
                self._display.blit(self._surface, (0, 0))
            pygame.display.flip()

        pygame.quit()


class HumanAgent(AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.agent_engaged = False
        self.current_control = carla.VehicleControl()
        self.current_control.steer = 0.0
        self.current_control.throttle = 1.0
        self.current_control.brake = 0.0
        self.current_control.hand_brake = False
        self._hic = HumanInterface(self)
        self._thread = Thread(target=self._hic.run)
        self._thread.start()

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor01'],
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor02'],

            ['sensor.lidar.ray_cast', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                       'yaw': yaw, 'pitch': pitch, 'roll': roll}, 'Sensor03']
        ]

        """
        sensors = [{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 300, 'height': 200, 'fov': 100, 'id': 'Center'},

                   {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    'yaw': -45.0, 'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

                   {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                    'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

                   {'type': 'sensor.camera.rgb', 'x': -1.8, 'y': 0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0,
                    'yaw': 180.0, 'width': 300, 'height': 200, 'fov': 130, 'id': 'Rear'},

                   {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
                   ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        self.agent_engaged = True
        time.sleep(0.1)
        return self.current_control

    def destroy(self):
        """
        Cleanup
        """
        self._hic.quit = True
        self._thread.join()


class KeyboardControl(object):

    """
    Keyboard control for the human agent
    """

    def __init__(self):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

    def parse_events(self, control, clock):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            control.steer = self._control.steer
            control.throttle = self._control.throttle
            control.brake = self._control.brake
            control.hand_brake = self._control.hand_brake

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        self._control.throttle = 0.6 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 15.0 * 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]
