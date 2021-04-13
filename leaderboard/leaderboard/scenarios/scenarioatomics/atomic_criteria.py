#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides additional atomic evaluation criteria required
to analyze if a scenario was completed successfully or failed.

Criteria should run continuously to monitor the state of a single actor, multiple
actors or environmental parameters. Hence, a termination is not required.

The atomic criteria are implemented with py_trees.
"""

import py_trees
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenariomanager.traffic_events import TrafficEvent, TrafficEventType


class ActorSpeedAboveThresholdTest(Criterion):
    """
    This test will fail if the actor has had its linear velocity lower than a specific value for
    a specific amount of time

    Important parameters:
    - actor: CARLA actor to be used for this test
    - speed_threshold: speed required
    - below_threshold_max_time: Maximum time (in seconds) the actor can remain under the speed threshold
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    def __init__(self, actor, speed_threshold, below_threshold_max_time,
                 name="ActorSpeedAboveThresholdTest", terminate_on_failure=False):
        """
        Class constructor.
        """
        super(ActorSpeedAboveThresholdTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self._actor = actor
        self._speed_threshold = speed_threshold
        self._below_threshold_max_time = below_threshold_max_time
        self._time_last_valid_state = None

    def update(self):
        """
        Check if the actor speed is above the speed_threshold
        """
        new_status = py_trees.common.Status.RUNNING

        linear_speed = CarlaDataProvider.get_velocity(self._actor)
        if linear_speed is not None:
            if linear_speed < self._speed_threshold and self._time_last_valid_state:
                if (GameTime.get_time() - self._time_last_valid_state) > self._below_threshold_max_time:
                    # Game over. The actor has been "blocked" for too long
                    self.test_status = "FAILURE"

                    # record event
                    vehicle_location = CarlaDataProvider.get_location(self._actor)
                    blocked_event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED)
                    ActorSpeedAboveThresholdTest._set_event_message(blocked_event, vehicle_location)
                    ActorSpeedAboveThresholdTest._set_event_dict(blocked_event, vehicle_location)
                    self.list_traffic_events.append(blocked_event)
            else:
                self._time_last_valid_state = GameTime.get_time()

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = py_trees.common.Status.FAILURE
        self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    @staticmethod
    def _set_event_message(event, location):
        """
        Sets the message of the event
        """

        event.set_message('Agent got blocked at (x={}, y={}, z={})'.format(round(location.x, 3),
                                                                               round(location.y, 3),
                                                                               round(location.z, 3)))
    @staticmethod
    def _set_event_dict(event, location):
        """
        Sets the dictionary of the event
        """
        event.set_dict({
            'x': location.x,
            'y': location.y,
            'z': location.z,
          })
