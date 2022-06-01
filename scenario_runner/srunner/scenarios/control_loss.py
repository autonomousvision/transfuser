#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Control Loss Vehicle scenario:

The scenario realizes that the vehicle looses control due to
bad road conditions, etc. and checks to see if the vehicle
regains control and corrects it's course.
"""

import numpy.random as random
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeNoiseParameters, ActorTransformSetter
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


class ControlLoss(BasicScenario):

    """
    Implementation of "Control Loss Vehicle" (Traffic Scenario 01)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        # ego vehicle parameters
        self._no_of_jitter = 10
        self._noise_mean = 0      # Mean value of steering noise
        self._noise_std = 0.01   # Std. deviation of steering noise
        self._dynamic_mean_for_steer = 0.001
        self._dynamic_mean_for_throttle = 0.045
        self._abort_distance_to_intersection = 10
        self._current_steer_noise = [0]  # This is a list, since lists are mutable
        self._current_throttle_noise = [0]
        self._start_distance = 20
        self._trigger_dist = 2
        self._end_distance = 30
        self._ego_vehicle_max_steer = 0.0
        self._ego_vehicle_max_throttle = 1.0
        self._ego_vehicle_target_velocity = 15
        self._map = CarlaDataProvider.get_map()
        # Timeout of scenario in seconds
        self.timeout = timeout
        # The reference trigger for the control loss
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self.loc_list = []
        self.obj = []
        self._randomize = randomize
        super(ControlLoss, self).__init__("ControlLoss",
                                          ego_vehicles,
                                          config,
                                          world,
                                          debug_mode,
                                          criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        if self._randomize:
            self._distance = random.randint(low=10, high=80, size=3)
            self._distance = sorted(self._distance)
        else:
            self._distance = [14, 48, 74]
        first_loc, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._distance[0])
        second_loc, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._distance[1])
        third_loc, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._distance[2])

        self.loc_list.extend([first_loc, second_loc, third_loc])
        self._dist_prop = [x - 2 for x in self._distance]

        self.first_loc_prev, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._dist_prop[0])
        self.sec_loc_prev, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._dist_prop[1])
        self.third_loc_prev, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._dist_prop[2])

        self.first_transform = carla.Transform(self.first_loc_prev)
        self.sec_transform = carla.Transform(self.sec_loc_prev)
        self.third_transform = carla.Transform(self.third_loc_prev)
        self.first_transform = carla.Transform(carla.Location(self.first_loc_prev.x,
                                                              self.first_loc_prev.y,
                                                              self.first_loc_prev.z))
        self.sec_transform = carla.Transform(carla.Location(self.sec_loc_prev.x,
                                                            self.sec_loc_prev.y,
                                                            self.sec_loc_prev.z))
        self.third_transform = carla.Transform(carla.Location(self.third_loc_prev.x,
                                                              self.third_loc_prev.y,
                                                              self.third_loc_prev.z))

        first_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.first_transform, 'prop')
        second_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.sec_transform, 'prop')
        third_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.third_transform, 'prop')

        first_debris.set_transform(self.first_transform)
        second_debris.set_transform(self.sec_transform)
        third_debris.set_transform(self.third_transform)

        self.obj.extend([first_debris, second_debris, third_debris])
        for debris in self.obj:
            debris.set_simulate_physics(False)

        self.other_actors.append(first_debris)
        self.other_actors.append(second_debris)
        self.other_actors.append(third_debris)

    def _create_behavior(self):
        """
        The scenario defined after is a "control loss vehicle" scenario. After
        invoking this scenario, it will wait until the vehicle drove a few meters
        (_start_distance), and then perform a jitter action. Finally, the vehicle
        has to reach a target point (_end_distance). If this does not happen within
        60 seconds, a timeout stops the scenario
        """
        # start condition
        start_end_parallel = py_trees.composites.Parallel("Jitter",
                                                          policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        start_condition = InTriggerDistanceToLocation(self.ego_vehicles[0], self.first_loc_prev, self._trigger_dist)
        for _ in range(self._no_of_jitter):

            # change the current noise to be applied
            turn = ChangeNoiseParameters(self._current_steer_noise, self._current_throttle_noise,
                                         self._noise_mean, self._noise_std, self._dynamic_mean_for_steer,
                                         self._dynamic_mean_for_throttle)  # Mean value of steering noise
        # Noise end! put again the added noise to zero.
        noise_end = ChangeNoiseParameters(self._current_steer_noise, self._current_throttle_noise,
                                          0, 0, 0, 0)

        jitter_action = py_trees.composites.Parallel("Jitter",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # Abort jitter_sequence, if the vehicle is approaching an intersection
        jitter_abort = InTriggerDistanceToNextIntersection(self.ego_vehicles[0], self._abort_distance_to_intersection)
        # endcondition: Check if vehicle reached waypoint _end_distance from here:
        end_condition = DriveDistance(self.ego_vehicles[0], self._end_distance)
        start_end_parallel.add_child(start_condition)
        start_end_parallel.add_child(end_condition)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("ControlLoss")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self.first_transform, physics=False))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self.sec_transform, physics=False))
        sequence.add_child(ActorTransformSetter(self.other_actors[2], self.third_transform, physics=False))
        jitter = py_trees.composites.Sequence("Jitter Behavior")
        jitter.add_child(turn)
        jitter.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], self.sec_loc_prev, self._trigger_dist))
        jitter.add_child(turn)
        jitter.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], self.third_loc_prev, self._trigger_dist))
        jitter.add_child(turn)
        jitter_action.add_child(jitter)
        jitter_action.add_child(jitter_abort)
        sequence.add_child(start_end_parallel)
        sequence.add_child(jitter_action)
        sequence.add_child(end_condition)
        sequence.add_child(noise_end)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def change_control(self, control):
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.
        """
        control.steer += self._current_steer_noise[0]
        control.throttle += self._current_throttle_noise[0]

        return control

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
