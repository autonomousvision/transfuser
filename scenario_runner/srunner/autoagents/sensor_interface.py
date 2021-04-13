#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This file containts CallBack class and SensorInterface, responsible of
handling the use of sensors for the agents
"""

import copy
import logging
import numpy as np

import carla


class CallBack(object):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor, data_provider):
        """
        Initializes the call back
        """
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor)

    def __call__(self, data):
        """
        call function
        """
        if isinstance(data, carla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        """
        parses cameras
        """
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        """
        parses lidar sensors
        """
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        """
        parses gnss sensors
        """
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)


class SensorInterface(object):

    """
    Class that contains all sensor data
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._sensors_objects = {}
        self._data_buffers = {}
        self._timestamps = {}

    def register_sensor(self, tag, sensor):
        """
        Registers the sensors
        """
        if tag in self._sensors_objects:
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        self._sensors_objects[tag] = sensor
        self._data_buffers[tag] = None
        self._timestamps[tag] = -1

    def update_sensor(self, tag, data, timestamp):
        """
        Updates the sensor
        """
        if tag not in self._sensors_objects:
            raise ValueError("The sensor with tag [{}] has not been created!".format(tag))
        self._data_buffers[tag] = data
        self._timestamps[tag] = timestamp

    def all_sensors_ready(self):
        """
        Checks if all the sensors have sent data at least once
        """
        for key in self._sensors_objects:
            if self._data_buffers[key] is None:
                return False
        return True

    def get_data(self):
        """
        Returns the data of a sensor
        """
        data_dict = {}
        for key in self._sensors_objects:
            data_dict[key] = (self._timestamps[key], self._data_buffers[key])
        return data_dict
