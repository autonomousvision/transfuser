#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Summary of useful helper functions for scenarios
"""

import math
import shapely.geometry
import shapely.affinity

import numpy as np

import carla
from agents.tools.misc import vector
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_distance_along_route(route, target_location):
    """
    Calculate the distance of the given location along the route

    Note: If the location is not along the route, the route length will be returned
    """

    wmap = CarlaDataProvider.get_map()
    covered_distance = 0
    prev_position = None
    found = False

    # Don't use the input location, use the corresponding wp as location
    target_location_from_wp = wmap.get_waypoint(target_location).transform.location

    for position, _ in route:

        location = target_location_from_wp

        # Don't perform any calculations for the first route point
        if not prev_position:
            prev_position = position
            continue

        # Calculate distance between previous and current route point
        interval_length_squared = ((prev_position.x - position.x) ** 2) + ((prev_position.y - position.y) ** 2)
        distance_squared = ((location.x - prev_position.x) ** 2) + ((location.y - prev_position.y) ** 2)

        # Close to the current position? Stop calculation
        if distance_squared < 0.01:
            break

        if distance_squared < 400 and not distance_squared < interval_length_squared:
            # Check if a neighbor lane is closer to the route
            # Do this only in a close distance to correct route interval, otherwise the computation load is too high
            starting_wp = wmap.get_waypoint(location)
            wp = starting_wp.get_left_lane()
            while wp is not None:
                new_location = wp.transform.location
                new_distance_squared = ((new_location.x - prev_position.x) ** 2) + (
                    (new_location.y - prev_position.y) ** 2)

                if np.sign(starting_wp.lane_id) != np.sign(wp.lane_id):
                    break

                if new_distance_squared < distance_squared:
                    distance_squared = new_distance_squared
                    location = new_location
                else:
                    break

                wp = wp.get_left_lane()

            wp = starting_wp.get_right_lane()
            while wp is not None:
                new_location = wp.transform.location
                new_distance_squared = ((new_location.x - prev_position.x) ** 2) + (
                    (new_location.y - prev_position.y) ** 2)

                if np.sign(starting_wp.lane_id) != np.sign(wp.lane_id):
                    break

                if new_distance_squared < distance_squared:
                    distance_squared = new_distance_squared
                    location = new_location
                else:
                    break

                wp = wp.get_right_lane()

        if distance_squared < interval_length_squared:
            # The location could be inside the current route interval, if route/lane ids match
            # Note: This assumes a sufficiently small route interval
            # An alternative is to compare orientations, however, this also does not work for
            # long route intervals

            curr_wp = wmap.get_waypoint(position)
            prev_wp = wmap.get_waypoint(prev_position)
            wp = wmap.get_waypoint(location)

            if prev_wp and curr_wp and wp:
                if wp.road_id == prev_wp.road_id or wp.road_id == curr_wp.road_id:
                    # Roads match, now compare the sign of the lane ids
                    if (np.sign(wp.lane_id) == np.sign(prev_wp.lane_id) or
                            np.sign(wp.lane_id) == np.sign(curr_wp.lane_id)):
                        # The location is within the current route interval
                        covered_distance += math.sqrt(distance_squared)
                        found = True
                        break

        covered_distance += math.sqrt(interval_length_squared)
        prev_position = position

    return covered_distance, found


def get_crossing_point(actor):
    """
    Get the next crossing point location in front of the ego vehicle

    @return point of crossing
    """
    wp_cross = CarlaDataProvider.get_map().get_waypoint(actor.get_location())

    while not wp_cross.is_intersection:
        wp_cross = wp_cross.next(2)[0]

    crossing = carla.Location(x=wp_cross.transform.location.x,
                              y=wp_cross.transform.location.y, z=wp_cross.transform.location.z)

    return crossing


def get_geometric_linear_intersection(ego_actor, other_actor):
    """
    Obtain a intersection point between two actor's location by using their waypoints (wp)

    @return point of intersection of the two vehicles
    """

    wp_ego_1 = CarlaDataProvider.get_map().get_waypoint(ego_actor.get_location())
    wp_ego_2 = wp_ego_1.next(1)[0]
    x_ego_1 = wp_ego_1.transform.location.x
    y_ego_1 = wp_ego_1.transform.location.y
    x_ego_2 = wp_ego_2.transform.location.x
    y_ego_2 = wp_ego_2.transform.location.y

    wp_other_1 = CarlaDataProvider.get_world().get_map().get_waypoint(other_actor.get_location())
    wp_other_2 = wp_other_1.next(1)[0]
    x_other_1 = wp_other_1.transform.location.x
    y_other_1 = wp_other_1.transform.location.y
    x_other_2 = wp_other_2.transform.location.x
    y_other_2 = wp_other_2.transform.location.y

    s = np.vstack([(x_ego_1, y_ego_1), (x_ego_2, y_ego_2), (x_other_1, y_other_1), (x_other_2, y_other_2)])
    h = np.hstack((s, np.ones((4, 1))))
    line1 = np.cross(h[0], h[1])
    line2 = np.cross(h[2], h[3])
    x, y, z = np.cross(line1, line2)
    if z == 0:
        return (float('inf'), float('inf'))

    intersection = carla.Location(x=x / z, y=y / z, z=0)

    return intersection


def get_location_in_distance(actor, distance):
    """
    Obtain a location in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.

    @return obtained location and the traveled distance
    """
    waypoint = CarlaDataProvider.get_map().get_waypoint(actor.get_location())
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint.transform.location, traveled_distance


def get_location_in_distance_from_wp(waypoint, distance, stop_at_junction=True):
    """
    Obtain a location in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.

    @return obtained location and the traveled distance
    """
    traveled_distance = 0
    while not (waypoint.is_intersection and stop_at_junction) and traveled_distance < distance:
        wp_next = waypoint.next(1.0)
        if wp_next:
            waypoint_new = wp_next[-1]
            traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
            waypoint = waypoint_new
        else:
            break

    return waypoint.transform.location, traveled_distance


def get_waypoint_in_distance(waypoint, distance):
    """
    Obtain a waypoint in a given distance from the current actor's location.
    Note: Search is stopped on first intersection.
    @return obtained waypoint and the traveled distance
    """
    traveled_distance = 0
    while not waypoint.is_intersection and traveled_distance < distance:
        waypoint_new = waypoint.next(1.0)[-1]
        traveled_distance += waypoint_new.transform.location.distance(waypoint.transform.location)
        waypoint = waypoint_new

    return waypoint, traveled_distance


def generate_target_waypoint_list(waypoint, turn=0):
    """
    This method follow waypoints to a junction and choose path based on turn input.
    Turn input: LEFT -> -1, RIGHT -> 1, STRAIGHT -> 0
    @returns a waypoint list from the starting point to the end point according to turn input
    """
    reached_junction = False
    threshold = math.radians(0.1)
    plan = []
    while True:
        wp_choice = waypoint.next(2)
        if len(wp_choice) > 1:
            reached_junction = True
            waypoint = choose_at_junction(waypoint, wp_choice, turn)
        else:
            waypoint = wp_choice[0]
        plan.append((waypoint, RoadOption.LANEFOLLOW))
        #   End condition for the behavior
        if turn != 0 and reached_junction and len(plan) >= 3:
            v_1 = vector(
                plan[-2][0].transform.location,
                plan[-1][0].transform.location)
            v_2 = vector(
                plan[-3][0].transform.location,
                plan[-2][0].transform.location)
            angle_wp = math.acos(
                np.dot(v_1, v_2) / abs((np.linalg.norm(v_1) * np.linalg.norm(v_2))))
            if angle_wp < threshold:
                break
        elif reached_junction and not plan[-1][0].is_intersection:
            break

    return plan, plan[-1][0]


def generate_target_waypoint_list_multilane(waypoint, change='left',
                                            distance_same_lane=10,
                                            distance_other_lane=25,
                                            total_lane_change_distance=25, check='true'):
    """
    This methods generates a waypoint list which leads the vehicle to a parallel lane.
    The change input must be 'left' or 'right', depending on which lane you want to change.

    The step distance between waypoints on the same lane is 2m.
    The step distance between the lane change is set to 25m.

    @returns a waypoint list from the starting point to the end point on a right or left parallel lane.
    """
    plan = []
    plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

    step_distance = 2

    # check if lane change possible
    if check == 'true':
        lane_change_possibilities = ['Left', 'Right', 'Both']
        if str(waypoint.lane_change) not in lane_change_possibilities:
            # ERROR, lane change is not possible
            return None

    # same lane
    distance = 0
    while distance < distance_same_lane:
        next_wp = plan[-1][0].next(step_distance)
        distance += next_wp[0].transform.location.distance(plan[-1][0].transform.location)
        plan.append((next_wp[0], RoadOption.LANEFOLLOW))

    target_lane_id = None
    if change == 'left':
        # go left
        wp_left = plan[-1][0].get_left_lane()
        target_lane_id = wp_left.lane_id
        next_wp = wp_left.next(total_lane_change_distance)
        plan.append((next_wp[0], RoadOption.LANEFOLLOW))
    elif change == 'right':
        # go right
        wp_right = plan[-1][0].get_right_lane()
        target_lane_id = wp_right.lane_id
        next_wp = wp_right.next(total_lane_change_distance)
        plan.append((next_wp[0], RoadOption.LANEFOLLOW))
    else:
        # ERROR, input value for change must be 'left' or 'right'
        return None

    # other lane
    distance = 0
    while distance < distance_other_lane:
        next_wp = plan[-1][0].next(step_distance)
        distance += next_wp[0].transform.location.distance(plan[-1][0].transform.location)
        plan.append((next_wp[0], RoadOption.LANEFOLLOW))

    return plan, target_lane_id


def generate_target_waypoint(waypoint, turn=0):
    """
    This method follow waypoints to a junction and choose path based on turn input.
    Turn input: LEFT -> -1, RIGHT -> 1, STRAIGHT -> 0
    @returns a waypoint list according to turn input
    """
    sampling_radius = 1
    reached_junction = False
    wp_list = []
    while True:

        wp_choice = waypoint.next(sampling_radius)
        #   Choose path at intersection
        if not reached_junction and (len(wp_choice) > 1 or wp_choice[0].is_junction):
            reached_junction = True
            waypoint = choose_at_junction(waypoint, wp_choice, turn)
        else:
            waypoint = wp_choice[0]
        wp_list.append(waypoint)
        #   End condition for the behavior
        if reached_junction and not wp_list[-1].is_junction:
            break
    return wp_list[-1]


def generate_target_waypoint_in_route(waypoint, route):
    """
    This method follow waypoints to a junction
    @returns a waypoint list according to turn input
    """
    wmap = CarlaDataProvider.get_map()
    reached_junction = False

    # Get the route location
    shortest_distance = float('inf')
    for index, route_pos in enumerate(route):
        wp = route_pos[0]
        trigger_location = waypoint.transform.location

        dist_to_route = trigger_location.distance(wp)
        if dist_to_route <= shortest_distance:
            closest_index = index
            shortest_distance = dist_to_route

    route_location = route[closest_index][0]
    index = closest_index

    while True:
        # Get the next route location
        index = min(index + 1, len(route))
        route_location = route[index][0]
        road_option = route[index][1]

        # Enter the junction
        if not reached_junction and (road_option in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT)):
            reached_junction = True

        # End condition for the behavior, at the end of the junction
        if reached_junction and (road_option not in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT)):
            break

    return wmap.get_waypoint(route_location)


def choose_at_junction(current_waypoint, next_choices, direction=0):
    """
    This function chooses the appropriate waypoint from next_choices based on direction
    """
    current_transform = current_waypoint.transform
    current_location = current_transform.location
    projected_location = current_location + \
        carla.Location(
            x=math.cos(math.radians(current_transform.rotation.yaw)),
            y=math.sin(math.radians(current_transform.rotation.yaw)))
    current_vector = vector(current_location, projected_location)
    cross_list = []
    cross_to_waypoint = dict()
    for waypoint in next_choices:
        waypoint = waypoint.next(10)[0]
        select_vector = vector(current_location, waypoint.transform.location)
        cross = np.cross(current_vector, select_vector)[2]
        cross_list.append(cross)
        cross_to_waypoint[cross] = waypoint
    select_cross = None
    if direction > 0:
        select_cross = max(cross_list)
    elif direction < 0:
        select_cross = min(cross_list)
    else:
        select_cross = min(cross_list, key=abs)

    return cross_to_waypoint[select_cross]


def get_intersection(ego_actor, other_actor):
    """
    Obtain a intersection point between two actor's location
    @return the intersection location
    """
    waypoint = CarlaDataProvider.get_map().get_waypoint(ego_actor.get_location())
    waypoint_other = CarlaDataProvider.get_map().get_waypoint(other_actor.get_location())
    max_dist = float("inf")
    distance = float("inf")
    while distance <= max_dist:
        max_dist = distance
        current_location = waypoint.transform.location
        waypoint_choice = waypoint.next(1)
        #   Select the straighter path at intersection
        if len(waypoint_choice) > 1:
            max_dot = -1 * float('inf')
            loc_projection = current_location + carla.Location(
                x=math.cos(math.radians(waypoint.transform.rotation.yaw)),
                y=math.sin(math.radians(waypoint.transform.rotation.yaw)))
            v_current = vector(current_location, loc_projection)
            for wp_select in waypoint_choice:
                v_select = vector(current_location, wp_select.transform.location)
                dot_select = np.dot(v_current, v_select)
                if dot_select > max_dot:
                    max_dot = dot_select
                    waypoint = wp_select
        else:
            waypoint = waypoint_choice[0]
        distance = current_location.distance(waypoint_other.transform.location)

    return current_location


def detect_lane_obstacle(actor, extension_factor=3, margin=1.02):
    """
    This function identifies if an obstacle is present in front of the reference actor
    """
    world = CarlaDataProvider.get_world()
    world_actors = world.get_actors().filter('vehicle.*')
    actor_bbox = actor.bounding_box
    actor_transform = actor.get_transform()
    actor_location = actor_transform.location
    actor_vector = actor_transform.rotation.get_forward_vector()
    actor_vector = np.array([actor_vector.x, actor_vector.y])
    actor_vector = actor_vector / np.linalg.norm(actor_vector)
    actor_vector = actor_vector * (extension_factor - 1) * actor_bbox.extent.x
    actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
    actor_yaw = actor_transform.rotation.yaw

    is_hazard = False
    for adversary in world_actors:
        if adversary.id != actor.id and \
                actor_transform.location.distance(adversary.get_location()) < 50:
            adversary_bbox = adversary.bounding_box
            adversary_transform = adversary.get_transform()
            adversary_loc = adversary_transform.location
            adversary_yaw = adversary_transform.rotation.yaw
            overlap_adversary = RotatedRectangle(
                adversary_loc.x, adversary_loc.y,
                2 * margin * adversary_bbox.extent.x, 2 * margin * adversary_bbox.extent.y, adversary_yaw)
            overlap_actor = RotatedRectangle(
                actor_location.x, actor_location.y,
                2 * margin * actor_bbox.extent.x * extension_factor, 2 * margin * actor_bbox.extent.y, actor_yaw)
            overlap_area = overlap_adversary.intersection(overlap_actor).area
            if overlap_area > 0:
                is_hazard = True
                break

    return is_hazard


class RotatedRectangle(object):

    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width      # pylint: disable=invalid-name
        self.h = height     # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())
