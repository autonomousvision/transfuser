#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function
import sys

import py_trees
import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      WaypointFollower,
                                                                      SyncArrival)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, DrivenDistanceTest, MaxVelocityTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (get_crossing_point,
                                           get_geometric_linear_intersection,
                                           generate_target_waypoint_list)


class OppositeVehicleRunningRedLight(BasicScenario):

    """
    This class holds everything required for a scenario,
    in which an other vehicle takes priority from the ego
    vehicle, by running a red traffic light (while the ego
    vehicle has green) (Traffic Scenario 7)

    This is a single ego vehicle scenario
    """

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20       # Maximum allowed velocity [m/s]
    _ego_avg_velocity_expected = 4       # Average expected velocity [m/s]
    _ego_expected_driven_distance = 70   # Expected driven distance [m]
    _ego_distance_to_traffic_light = 32  # Trigger distance to traffic light [m]
    _ego_distance_to_drive = 40          # Allowed distance to drive

    # other vehicle
    _other_actor_target_velocity = 10      # Target velocity of other vehicle
    _other_actor_max_brake = 1.0           # Maximum brake of other vehicle
    _other_actor_distance = 50             # Distance the other vehicle should drive

    _traffic_light = None

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        self._other_actor_transform = None

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight",
                                                             ego_vehicles,
                                                             config,
                                                             world,
                                                             debug_mode,
                                                             criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)

        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)

        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)

        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)

        traffic_light_other.set_state(carla.TrafficLightState.Red)
        traffic_light_other.set_red_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle waits until the ego vehicle is close enough to the
        intersection and that its own traffic light is red. Then, it will start
        driving and 'illegally' cross the intersection. After a short distance
        it should stop again, outside of the intersection. The ego vehicle has
        to avoid the crash, but continue driving after the intersection is clear.

        If this does not happen within 120 seconds, a timeout stops the scenario
        """
        crossing_point_dynamic = get_crossing_point(self.ego_vehicles[0])

        # start condition
        startcondition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            crossing_point_dynamic,
            self._ego_distance_to_traffic_light,
            name="Waiting for start position")

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])

        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToNextIntersection(self.other_actors[0],
                                                                5)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

        # Generate plan for WaypointFollower
        turn = 0  # drive straight ahead
        plan = []

        # generating waypoints until intersection (target_waypoint)
        plan, target_waypoint = generate_target_waypoint_list(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), turn)

        # Generating waypoint list till next intersection
        wp_choice = target_waypoint.next(5.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(5.0)

        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        continue_driving_waypoints = WaypointFollower(
            self.other_actors[0], self._other_actor_target_velocity, plan=plan, avoid_collision=False)

        continue_driving_distance = DriveDistance(
            self.other_actors[0],
            self._other_actor_distance,
            name="Distance")

        continue_driving_timeout = TimeOut(10)

        continue_driving.add_child(continue_driving_waypoints)
        continue_driving.add_child(continue_driving_distance)
        continue_driving.add_child(continue_driving_timeout)

        # finally wait that ego vehicle drove a specific distance
        wait = DriveDistance(
            self.ego_vehicles[0],
            self._ego_distance_to_drive,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(startcondition)
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(continue_driving)
        sequence.add_child(wait)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicles[0],
            self._ego_max_velocity_allowed,
            optional=True)
        collision_criterion = CollisionTest(self.ego_vehicles[0])
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicles[0],
            self._ego_expected_driven_distance)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)

        # Add the collision and lane checks for all vehicles as well
        for vehicle in self.other_actors:
            collision_criterion = CollisionTest(vehicle)
            criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self._traffic_light = None
        self.remove_all_actors()
