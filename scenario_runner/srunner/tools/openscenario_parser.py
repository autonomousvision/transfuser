#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a parser for scenario configuration files based on OpenSCENARIO
"""

from distutils.util import strtobool
import copy
import datetime
import math
import operator

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.weather_sim import Weather
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (TrafficLightStateSetter,
                                                                      ActorTransformSetterToOSCPosition,
                                                                      RunScript,
                                                                      ChangeWeather,
                                                                      ChangeAutoPilot,
                                                                      ChangeRoadFriction,
                                                                      ChangeActorTargetSpeed,
                                                                      ChangeActorControl,
                                                                      ChangeActorWaypoints,
                                                                      ChangeActorWaypointsToReachPosition,
                                                                      ChangeActorLateralMotion,
                                                                      Idle)
# pylint: disable=unused-import
# For the following includes the pylint check is disabled, as these are accessed via globals()
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     MaxVelocityTest,
                                                                     DrivenDistanceTest,
                                                                     AverageVelocityTest,
                                                                     KeepLaneTest,
                                                                     ReachedRegionTest,
                                                                     OnSidewalkTest,
                                                                     WrongLaneTest,
                                                                     InRadiusRegionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     OffRoadTest,
                                                                     EndofRoadTest)
# pylint: enable=unused-import
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToOSCPosition,
                                                                               InTimeToArrivalToOSCPosition,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance,
                                                                               StandStill,
                                                                               OSCStartEndCondition,
                                                                               TriggerAcceleration,
                                                                               RelativeVelocityToOtherActor,
                                                                               TimeOfDayComparison,
                                                                               TriggerVelocity,
                                                                               WaitForTrafficLightState)
from srunner.scenariomanager.timer import TimeOut, SimulationTimeCondition
from srunner.tools.py_trees_port import oneshot_behavior


class OpenScenarioParser(object):

    """
    Pure static class providing conversions from OpenSCENARIO elements to ScenarioRunner elements
    """
    operators = {
        "greaterThan": operator.gt,
        "lessThan": operator.lt,
        "equalTo": operator.eq
    }

    actor_types = {
        "pedestrian": "walker",
        "vehicle": "vehicle",
        "miscellaneous": "miscellaneous"
    }

    tl_states = {
        "GREEN": carla.TrafficLightState.Green,
        "YELLOW": carla.TrafficLightState.Yellow,
        "RED": carla.TrafficLightState.Red,
        "OFF": carla.TrafficLightState.Off,
    }

    global_osc_parameters = dict()
    use_carla_coordinate_system = False
    osc_filepath = None

    @staticmethod
    def get_traffic_light_from_osc_name(name):
        """
        Returns a carla.TrafficLight instance that matches the name given
        """
        traffic_light = None

        # Given by id
        if name.startswith("id="):
            tl_id = name[3:]
            for carla_tl in CarlaDataProvider.get_world().get_actors().filter('traffic.traffic_light'):
                if carla_tl.id == tl_id:
                    traffic_light = carla_tl
                    break
        # Given by position
        elif name.startswith("pos="):
            tl_pos = name[4:]
            pos = tl_pos.split(",")
            for carla_tl in CarlaDataProvider.get_world().get_actors().filter('traffic.traffic_light'):
                carla_tl_location = carla_tl.get_transform().location
                distance = carla_tl_location.distance(carla.Location(float(pos[0]),
                                                                     float(pos[1]),
                                                                     carla_tl_location.z))
                if distance < 2.0:
                    traffic_light = carla_tl
                    break

        if traffic_light is None:
            raise AttributeError("Unknown  traffic light {}".format(name))

        return traffic_light

    @staticmethod
    def set_osc_filepath(filepath):
        """
        Set path of OSC file. This is required if for example custom commands are provided with
        relative paths.
        """
        OpenScenarioParser.osc_filepath = filepath

    @staticmethod
    def set_use_carla_coordinate_system():
        """
        CARLA internally uses a left-hand coordinate system (Unreal), but OpenSCENARIO and OpenDRIVE
        are intended for right-hand coordinate system. Hence, we need to invert the coordinates, if
        the scenario does not use CARLA coordinates, but instead right-hand coordinates.
        """
        OpenScenarioParser.use_carla_coordinate_system = True

    @staticmethod
    def set_parameters(xml_tree, additional_parameter_dict=None):
        """
        Parse the xml_tree, and replace all parameter references
        with the actual values.

        Note: Parameter names must not start with "$", however when referencing a parameter the
              reference has to start with "$".
              https://releases.asam.net/OpenSCENARIO/1.0.0/ASAM_OpenSCENARIO_BS-1-2_User-Guide_V1-0-0.html#_re_use_mechanisms

        Args:
            xml_tree: Containing all nodes that should be updated
            additional_parameter_dict (dictionary): Additional parameters as dict (key, value). Optional.

        returns:
            updated xml_tree, dictonary containing all parameters and their values
        """

        parameter_dict = dict()
        if additional_parameter_dict is not None:
            parameter_dict = additional_parameter_dict
        parameters = xml_tree.find('ParameterDeclarations')

        if parameters is None and not parameter_dict:
            return xml_tree, parameter_dict

        if parameters is None:
            parameters = []

        for parameter in parameters:
            name = parameter.attrib.get('name')
            value = parameter.attrib.get('value')
            parameter_dict[name] = value

        for node in xml_tree.iter():
            for key in node.attrib:
                for param in sorted(parameter_dict, key=len, reverse=True):
                    if "$" + param in node.attrib[key]:
                        node.attrib[key] = node.attrib[key].replace("$" + param, parameter_dict[param])

        return xml_tree, parameter_dict

    @staticmethod
    def set_global_parameters(parameter_dict):
        """
        Set global_osc_parameter dictionary

        Args:
            parameter_dict (Dictionary): Input for global_osc_parameter
        """
        OpenScenarioParser.global_osc_parameters = parameter_dict

    @staticmethod
    def get_catalog_entry(catalogs, catalog_reference):
        """
        Get catalog entry referenced by catalog_reference included correct parameter settings

        Args:
            catalogs (Dictionary of dictionaries): List of all catalogs and their entries
            catalog_reference (XML ElementTree): Reference containing the exact catalog to be used

        returns:
            Catalog entry (XML ElementTree)
        """

        entry = catalogs[catalog_reference.attrib.get("catalogName")][catalog_reference.attrib.get("entryName")]
        entry_copy = copy.deepcopy(entry)
        catalog_copy = copy.deepcopy(catalog_reference)
        entry = OpenScenarioParser.assign_catalog_parameters(entry_copy, catalog_copy)

        return entry

    @staticmethod
    def assign_catalog_parameters(entry_instance, catalog_reference):
        """
        Parse catalog_reference, and replace all parameter references
        in entry_instance by the values provided in catalog_reference.

        Not to be used from outside this class.

        Args:
            entry_instance (XML ElementTree): Entry to be updated
            catalog_reference (XML ElementTree): Reference containing the exact parameter values

        returns:
            updated entry_instance with updated parameter values
        """

        parameter_dict = dict()
        for elem in entry_instance.iter():
            if elem.find('ParameterDeclarations') is not None:
                parameters = elem.find('ParameterDeclarations')
                for parameter in parameters:
                    name = parameter.attrib.get('name')
                    value = parameter.attrib.get('value')
                    parameter_dict[name] = value

        for parameter_assignments in catalog_reference.iter("ParameterAssignments"):
            for parameter_assignment in parameter_assignments.iter("ParameterAssignment"):
                parameter = parameter_assignment.attrib.get("parameterRef")
                value = parameter_assignment.attrib.get("value")
                parameter_dict[parameter] = value

        for node in entry_instance.iter():
            for key in node.attrib:
                for param in sorted(parameter_dict, key=len, reverse=True):
                    if "$" + param in node.attrib[key]:
                        node.attrib[key] = node.attrib[key].replace("$" + param, parameter_dict[param])

        OpenScenarioParser.set_parameters(entry_instance, OpenScenarioParser.global_osc_parameters)

        return entry_instance

    @staticmethod
    def get_friction_from_env_action(xml_tree, catalogs):
        """
        Extract the CARLA road friction coefficient from an OSC EnvironmentAction

        Args:
            xml_tree: Containing the EnvironmentAction,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           friction (float)
        """
        set_environment = next(xml_tree.iter("EnvironmentAction"))

        if sum(1 for _ in set_environment.iter("Weather")) != 0:
            environment = set_environment.find("Environment")
        elif set_environment.find("CatalogReference") is not None:
            catalog_reference = set_environment.find("CatalogReference")
            environment = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference)

        friction = 1.0

        road_condition = environment.iter("RoadCondition")
        for condition in road_condition:
            friction = condition.attrib.get('frictionScaleFactor')

        return friction

    @staticmethod
    def get_weather_from_env_action(xml_tree, catalogs):
        """
        Extract the CARLA weather parameters from an OSC EnvironmentAction

        Args:
            xml_tree: Containing the EnvironmentAction,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           Weather (srunner.scenariomanager.weather_sim.Weather)
        """
        set_environment = next(xml_tree.iter("EnvironmentAction"))

        if sum(1 for _ in set_environment.iter("Weather")) != 0:
            environment = set_environment.find("Environment")
        elif set_environment.find("CatalogReference") is not None:
            catalog_reference = set_environment.find("CatalogReference")
            environment = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference)

        weather = environment.find("Weather")
        sun = weather.find("Sun")

        carla_weather = carla.WeatherParameters()
        carla_weather.sun_azimuth_angle = math.degrees(float(sun.attrib.get('azimuth', 0)))
        carla_weather.sun_altitude_angle = math.degrees(float(sun.attrib.get('elevation', 0)))
        carla_weather.cloudiness = 100 - float(sun.attrib.get('intensity', 0)) * 100
        fog = weather.find("Fog")
        carla_weather.fog_distance = float(fog.attrib.get('visualRange', 'inf'))
        if carla_weather.fog_distance < 1000:
            carla_weather.fog_density = 100
        carla_weather.precipitation = 0
        carla_weather.precipitation_deposits = 0
        carla_weather.wetness = 0
        carla_weather.wind_intensity = 0
        precepitation = weather.find("Precipitation")
        if precepitation.attrib.get('precipitationType') == "rain":
            carla_weather.precipitation = float(precepitation.attrib.get('intensity')) * 100
            carla_weather.precipitation_deposits = 100  # if it rains, make the road wet
            carla_weather.wetness = carla_weather.precipitation
        elif precepitation.attrib.get('type') == "snow":
            raise AttributeError("CARLA does not support snow precipitation")

        time_of_day = environment.find("TimeOfDay")
        weather_animation = strtobool(time_of_day.attrib.get("animation"))
        time = time_of_day.attrib.get("dateTime")
        dtime = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")

        return Weather(carla_weather, dtime, weather_animation)

    @staticmethod
    def get_controller(xml_tree, catalogs):
        """
        Extract the object controller from the OSC XML or a catalog

        Args:
            xml_tree: Containing the controller information,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the EnvironmentAction

        returns:
           module: Python module containing the controller implementation
           args: Dictonary with (key, value) parameters for the controller
        """

        assign_action = next(xml_tree.iter("AssignControllerAction"))

        properties = None
        if assign_action.find('Controller') is not None:
            properties = assign_action.find('Controller').find('Properties')
        elif assign_action.find("CatalogReference") is not None:
            catalog_reference = assign_action.find("CatalogReference")
            properties = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference).find('Properties')

        module = None
        args = {}
        for prop in properties:
            if prop.attrib.get('name') == "module":
                module = prop.attrib.get('value')
            else:
                args[prop.attrib.get('name')] = prop.attrib.get('value')

        override_action = xml_tree.find('OverrideControllerValueAction')
        for child in override_action:
            if strtobool(child.attrib.get('active')):
                raise NotImplementedError("Controller override actions are not yet supported")

        return module, args

    @staticmethod
    def get_route(xml_tree, catalogs):
        """
        Extract the route from the OSC XML or a catalog

        Args:
            xml_tree: Containing the route information,
                or the reference to the catalog it is defined in.
            catalogs: XML Catalogs that could contain the Route

        returns:
           waypoints: List of route waypoints
        """
        route = None

        if xml_tree.find('Route') is not None:
            route = xml_tree.find('Route')
        elif xml_tree.find('CatalogReference') is not None:
            catalog_reference = xml_tree.find("CatalogReference")
            route = OpenScenarioParser.get_catalog_entry(catalogs, catalog_reference)
        else:
            raise AttributeError("Unknown private FollowRoute action")

        waypoints = []
        if route is not None:
            for waypoint in route.iter('Waypoint'):
                position = waypoint.find('Position')
                transform = OpenScenarioParser.convert_position_to_transform(position)
                waypoints.append(transform)

        return waypoints

    @staticmethod
    def convert_position_to_transform(position, actor_list=None):
        """
        Convert an OpenScenario position into a CARLA transform

        Not supported: Road, RelativeRoad, Lane, RelativeLane as the PythonAPI currently
                       does not provide sufficient access to OpenDrive information
                       Also not supported is Route. This can be added by checking additional
                       route information
        """

        if position.find('WorldPosition') is not None:
            world_pos = position.find('WorldPosition')
            x = float(world_pos.attrib.get('x', 0))
            y = float(world_pos.attrib.get('y', 0))
            z = float(world_pos.attrib.get('z', 0))
            yaw = math.degrees(float(world_pos.attrib.get('h', 0)))
            pitch = math.degrees(float(world_pos.attrib.get('p', 0)))
            roll = math.degrees(float(world_pos.attrib.get('r', 0)))
            if not OpenScenarioParser.use_carla_coordinate_system:
                y = y * (-1.0)
                yaw = yaw * (-1.0)
            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))

        elif ((position.find('RelativeWorldPosition') is not None) or
              (position.find('RelativeObjectPosition') is not None) or
              (position.find('RelativeLanePosition') is not None)):

            if position.find('RelativeWorldPosition') is not None:
                rel_pos = position.find('RelativeWorldPosition')
            if position.find('RelativeObjectPosition') is not None:
                rel_pos = position.find('RelativeObjectPosition')
            if position.find('RelativeLanePosition') is not None:
                rel_pos = position.find('RelativeLanePosition')

            # get relative object and relative position
            obj = rel_pos.attrib.get('entityRef')
            obj_actor = None
            actor_transform = None

            if actor_list is not None:
                for actor in actor_list:
                    if actor.rolename == obj:
                        obj_actor = actor
                        actor_transform = actor.transform
            else:
                for actor in CarlaDataProvider.get_world().get_actors():
                    if 'role_name' in actor.attributes and actor.attributes['role_name'] == obj:
                        obj_actor = actor
                        actor_transform = obj_actor.get_transform()
                        break

            if obj_actor is None:
                raise AttributeError("Object '{}' provided as position reference is not known".format(obj))

            # calculate orientation h, p, r
            is_absolute = False
            dyaw = 0
            dpitch = 0
            droll = 0
            if rel_pos.find('Orientation') is not None:
                orientation = rel_pos.find('Orientation')
                is_absolute = (orientation.attrib.get('type') == "absolute")
                dyaw = math.degrees(float(orientation.attrib.get('h', 0)))
                dpitch = math.degrees(float(orientation.attrib.get('p', 0)))
                droll = math.degrees(float(orientation.attrib.get('r', 0)))

            if not OpenScenarioParser.use_carla_coordinate_system:
                dyaw = dyaw * (-1.0)

            yaw = actor_transform.rotation.yaw
            pitch = actor_transform.rotation.pitch
            roll = actor_transform.rotation.roll

            if not is_absolute:
                yaw = yaw + dyaw
                pitch = pitch + dpitch
                roll = roll + droll
            else:
                yaw = dyaw
                pitch = dpitch
                roll = droll

            # calculate location x, y, z
            # dx, dy, dz
            if ((position.find('RelativeWorldPosition') is not None) or
                    (position.find('RelativeObjectPosition') is not None)):
                dx = float(rel_pos.attrib.get('dx', 0))
                dy = float(rel_pos.attrib.get('dy', 0))
                dz = float(rel_pos.attrib.get('dz', 0))

                if not OpenScenarioParser.use_carla_coordinate_system:
                    dy = dy * (-1.0)

                x = actor_transform.location.x + dx
                y = actor_transform.location.y + dy
                z = actor_transform.location.z + dz

            # dLane, ds, offset
            elif position.find('RelativeLanePosition') is not None:
                dlane = float(rel_pos.attrib.get('dLane'))
                ds = float(rel_pos.attrib.get('ds'))
                offset = float(rel_pos.attrib.get('offset', 0.0))

                carla_map = CarlaDataProvider.get_map()
                relative_waypoint = carla_map.get_waypoint(actor_transform.location)

                if dlane == 0:
                    wp = relative_waypoint
                elif dlane == -1:
                    wp = relative_waypoint.get_left_lane()
                elif dlane == 1:
                    wp = relative_waypoint.get_right_lane()
                if wp is None:
                    raise AttributeError("Object '{}' position with dLane={} is not valid".format(obj, dlane))

                if ds < 0:
                    ds = (-1.0) * ds
                    wp = wp.previous(ds)[-1]
                else:
                    wp = wp.next(ds)[-1]

                # Adapt transform according to offset
                h = math.radians(wp.transform.rotation.yaw)
                x_offset = math.sin(h) * offset
                y_offset = math.cos(h) * offset

                if OpenScenarioParser.use_carla_coordinate_system:
                    x_offset = x_offset * (-1.0)
                    y_offset = y_offset * (-1.0)

                x = wp.transform.location.x + x_offset
                y = wp.transform.location.y + y_offset
                z = wp.transform.location.z

            return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))

        # Not implemented
        elif position.find('RoadPosition') is not None:
            raise NotImplementedError("Road positions are not yet supported")
        elif position.find('RelativeRoadPosition') is not None:
            raise NotImplementedError("RelativeRoad positions are not yet supported")
        elif position.find('LanePosition') is not None:
            lane_pos = position.find('LanePosition')
            road_id = int(lane_pos.attrib.get('roadId', 0))
            lane_id = int(lane_pos.attrib.get('laneId', 0))
            offset = float(lane_pos.attrib.get('offset', 0))
            s = float(lane_pos.attrib.get('s', 0))
            is_absolute = True
            waypoint = CarlaDataProvider.get_map().get_waypoint_xodr(road_id, lane_id, s)
            if waypoint is None:
                raise AttributeError("Lane position cannot be found")

            transform = waypoint.transform
            if lane_pos.find('Orientation') is not None:
                orientation = lane_pos.find('Orientation')
                dyaw = math.degrees(float(orientation.attrib.get('h', 0)))
                dpitch = math.degrees(float(orientation.attrib.get('p', 0)))
                droll = math.degrees(float(orientation.attrib.get('r', 0)))

                if not OpenScenarioParser.use_carla_coordinate_system:
                    dyaw = dyaw * (-1.0)

                transform.rotation.yaw = transform.rotation.yaw + dyaw
                transform.rotation.pitch = transform.rotation.pitch + dpitch
                transform.rotation.roll = transform.rotation.roll + droll

            if offset != 0:
                forward_vector = transform.rotation.get_forward_vector()
                orthogonal_vector = carla.Vector3D(x=-forward_vector.y, y=forward_vector.x, z=forward_vector.z)
                transform.location.x = transform.location.x + offset * orthogonal_vector.x
                transform.location.y = transform.location.y + offset * orthogonal_vector.y

            return transform
        elif position.find('RoutePosition') is not None:
            raise NotImplementedError("Route positions are not yet supported")
        else:
            raise AttributeError("Unknown position")

    @staticmethod
    def convert_condition_to_atomic(condition, actor_list):
        """
        Convert an OpenSCENARIO condition into a Behavior/Criterion atomic

        If there is a delay defined in the condition, then the condition is checked after the delay time
        passed by, e.g. <Condition name="" delay="5">.

        Note: Not all conditions are currently supported.
        """

        atomic = None
        delay_atomic = None
        condition_name = condition.attrib.get('name')

        if condition.attrib.get('delay') is not None and str(condition.attrib.get('delay')) != '0':
            delay = float(condition.attrib.get('delay'))
            delay_atomic = TimeOut(delay)

        if condition.find('ByEntityCondition') is not None:

            trigger_actor = None    # A-priori validation ensures that this will be not None
            triggered_actor = None

            for triggering_entities in condition.find('ByEntityCondition').iter('TriggeringEntities'):
                for entity in triggering_entities.iter('EntityRef'):
                    for actor in actor_list:
                        if entity.attrib.get('entityRef', None) == actor.attributes['role_name']:
                            trigger_actor = actor
                            break

            for entity_condition in condition.find('ByEntityCondition').iter('EntityCondition'):
                if entity_condition.find('EndOfRoadCondition') is not None:
                    end_road_condition = entity_condition.find('EndOfRoadCondition')
                    condition_duration = float(end_road_condition.attrib.get('duration'))
                    atomic_cls = py_trees.meta.inverter(EndofRoadTest)
                    atomic = atomic_cls(
                        trigger_actor, condition_duration, terminate_on_failure=True, name=condition_name)
                elif entity_condition.find('CollisionCondition') is not None:

                    collision_condition = entity_condition.find('CollisionCondition')

                    if collision_condition.find('EntityRef') is not None:
                        collision_entity = collision_condition.find('EntityRef')

                        for actor in actor_list:
                            if collision_entity.attrib.get('entityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break

                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                collision_condition.attrib.get('entityRef', None)))

                        atomic_cls = py_trees.meta.inverter(CollisionTest)
                        atomic = atomic_cls(trigger_actor, other_actor=triggered_actor,
                                            terminate_on_failure=True, name=condition_name)

                    elif collision_condition.find('ByType') is not None:
                        collision_type = collision_condition.find('ByType').attrib.get('type', None)

                        triggered_type = OpenScenarioParser.actor_types[collision_type]

                        atomic_cls = py_trees.meta.inverter(CollisionTest)
                        atomic = atomic_cls(trigger_actor, other_actor_type=triggered_type,
                                            terminate_on_failure=True, name=condition_name)

                    else:
                        atomic_cls = py_trees.meta.inverter(CollisionTest)
                        atomic = atomic_cls(trigger_actor, terminate_on_failure=True, name=condition_name)

                elif entity_condition.find('OffroadCondition') is not None:
                    off_condition = entity_condition.find('OffroadCondition')
                    condition_duration = float(off_condition.attrib.get('duration'))
                    atomic_cls = py_trees.meta.inverter(OffRoadTest)
                    atomic = atomic_cls(
                        trigger_actor, condition_duration, terminate_on_failure=True, name=condition_name)
                elif entity_condition.find('TimeHeadwayCondition') is not None:
                    headtime_condition = entity_condition.find('TimeHeadwayCondition')

                    condition_value = float(headtime_condition.attrib.get('value'))

                    condition_rule = headtime_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]

                    condition_freespace = strtobool(headtime_condition.attrib.get('freespace', False))
                    if condition_freespace:
                        raise NotImplementedError(
                            "TimeHeadwayCondition: freespace attribute is currently not implemented")
                    condition_along_route = strtobool(headtime_condition.attrib.get('alongRoute', False))

                    for actor in actor_list:
                        if headtime_condition.attrib.get('entityRef', None) == actor.attributes['role_name']:
                            triggered_actor = actor
                            break
                    if triggered_actor is None:
                        raise AttributeError("Cannot find actor '{}' for condition".format(
                            headtime_condition.attrib.get('entityRef', None)))

                    atomic = InTimeToArrivalToVehicle(
                        trigger_actor, triggered_actor, condition_value,
                        condition_along_route, condition_operator, condition_name
                    )

                elif entity_condition.find('TimeToCollisionCondition') is not None:
                    ttc_condition = entity_condition.find('TimeToCollisionCondition')

                    condition_rule = ttc_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]

                    condition_value = ttc_condition.attrib.get('value')
                    condition_target = ttc_condition.find('TimeToCollisionConditionTarget')

                    condition_freespace = strtobool(ttc_condition.attrib.get('freespace', False))
                    if condition_freespace:
                        raise NotImplementedError(
                            "TimeToCollisionCondition: freespace attribute is currently not implemented")
                    condition_along_route = strtobool(ttc_condition.attrib.get('alongRoute', False))

                    if condition_target.find('Position') is not None:
                        position = condition_target.find('Position')
                        atomic = InTimeToArrivalToOSCPosition(
                            trigger_actor, position, condition_value, condition_along_route, condition_operator)
                    else:
                        for actor in actor_list:
                            if ttc_condition.attrib.get('EntityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break
                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                ttc_condition.attrib.get('EntityRef', None)))

                        atomic = InTimeToArrivalToVehicle(
                            trigger_actor, triggered_actor, condition_value,
                            condition_along_route, condition_operator, condition_name)
                elif entity_condition.find('AccelerationCondition') is not None:
                    accel_condition = entity_condition.find('AccelerationCondition')
                    condition_value = float(accel_condition.attrib.get('value'))
                    condition_rule = accel_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]
                    atomic = TriggerAcceleration(
                        trigger_actor, condition_value, condition_operator, condition_name)
                elif entity_condition.find('StandStillCondition') is not None:
                    ss_condition = entity_condition.find('StandStillCondition')
                    duration = float(ss_condition.attrib.get('duration'))
                    atomic = StandStill(trigger_actor, condition_name, duration)
                elif entity_condition.find('SpeedCondition') is not None:
                    spd_condition = entity_condition.find('SpeedCondition')
                    condition_value = float(spd_condition.attrib.get('value'))
                    condition_rule = spd_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]

                    atomic = TriggerVelocity(
                        trigger_actor, condition_value, condition_operator, condition_name)
                elif entity_condition.find('RelativeSpeedCondition') is not None:
                    relspd_condition = entity_condition.find('RelativeSpeedCondition')
                    condition_value = float(relspd_condition.attrib.get('value'))
                    condition_rule = relspd_condition.attrib.get('rule')
                    condition_operator = OpenScenarioParser.operators[condition_rule]

                    for actor in actor_list:
                        if relspd_condition.attrib.get('entityRef', None) == actor.attributes['role_name']:
                            triggered_actor = actor
                            break

                    if triggered_actor is None:
                        raise AttributeError("Cannot find actor '{}' for condition".format(
                            relspd_condition.attrib.get('entityRef', None)))

                    atomic = RelativeVelocityToOtherActor(
                        trigger_actor, triggered_actor, condition_value, condition_operator, condition_name)
                elif entity_condition.find('TraveledDistanceCondition') is not None:
                    distance_condition = entity_condition.find('TraveledDistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))
                    atomic = DriveDistance(trigger_actor, distance_value, name=condition_name)
                elif entity_condition.find('ReachPositionCondition') is not None:
                    rp_condition = entity_condition.find('ReachPositionCondition')
                    distance_value = float(rp_condition.attrib.get('tolerance'))
                    position = rp_condition.find('Position')
                    atomic = InTriggerDistanceToOSCPosition(
                        trigger_actor, position, distance_value, name=condition_name)
                elif entity_condition.find('DistanceCondition') is not None:
                    distance_condition = entity_condition.find('DistanceCondition')

                    distance_value = float(distance_condition.attrib.get('value'))

                    distance_rule = distance_condition.attrib.get('rule')
                    distance_operator = OpenScenarioParser.operators[distance_rule]

                    distance_freespace = strtobool(distance_condition.attrib.get('freespace', False))
                    if distance_freespace:
                        raise NotImplementedError(
                            "DistanceCondition: freespace attribute is currently not implemented")
                    distance_along_route = strtobool(distance_condition.attrib.get('alongRoute', False))

                    if distance_condition.find('Position') is not None:
                        position = distance_condition.find('Position')
                        atomic = InTriggerDistanceToOSCPosition(
                            trigger_actor, position, distance_value, distance_along_route,
                            distance_operator, name=condition_name)

                elif entity_condition.find('RelativeDistanceCondition') is not None:
                    distance_condition = entity_condition.find('RelativeDistanceCondition')
                    distance_value = float(distance_condition.attrib.get('value'))

                    distance_freespace = strtobool(distance_condition.attrib.get('freespace', False))
                    if distance_freespace:
                        raise NotImplementedError(
                            "RelativeDistanceCondition: freespace attribute is currently not implemented")
                    if distance_condition.attrib.get('relativeDistanceType') == "cartesianDistance":
                        for actor in actor_list:
                            if distance_condition.attrib.get('entityRef', None) == actor.attributes['role_name']:
                                triggered_actor = actor
                                break

                        if triggered_actor is None:
                            raise AttributeError("Cannot find actor '{}' for condition".format(
                                distance_condition.attrib.get('entityRef', None)))

                        condition_rule = distance_condition.attrib.get('rule')
                        condition_operator = OpenScenarioParser.operators[condition_rule]
                        atomic = InTriggerDistanceToVehicle(
                            triggered_actor, trigger_actor, distance_value, condition_operator, name=condition_name)
                    else:
                        raise NotImplementedError(
                            "RelativeDistance condition with the given specification is not yet supported")
        elif condition.find('ByValueCondition') is not None:
            value_condition = condition.find('ByValueCondition')
            if value_condition.find('ParameterCondition') is not None:
                parameter_condition = value_condition.find('ParameterCondition')
                arg_name = parameter_condition.attrib.get('parameterRef')
                value = parameter_condition.attrib.get('value')
                if value != '':
                    arg_value = float(value)
                else:
                    arg_value = 0
                parameter_condition.attrib.get('rule')

                if condition_name in globals():
                    criterion_instance = globals()[condition_name]
                else:
                    raise AttributeError(
                        "The condition {} cannot be mapped to a criterion atomic".format(condition_name))

                atomic = py_trees.composites.Parallel("Evaluation Criteria for multiple ego vehicles")
                for triggered_actor in actor_list:
                    if arg_name != '':
                        atomic.add_child(criterion_instance(triggered_actor, arg_value))
                    else:
                        atomic.add_child(criterion_instance(triggered_actor))
            elif value_condition.find('SimulationTimeCondition') is not None:
                simtime_condition = value_condition.find('SimulationTimeCondition')
                value = float(simtime_condition.attrib.get('value'))
                rule = simtime_condition.attrib.get('rule')
                atomic = SimulationTimeCondition(value, success_rule=rule)
            elif value_condition.find('TimeOfDayCondition') is not None:
                tod_condition = value_condition.find('TimeOfDayCondition')
                condition_date = tod_condition.attrib.get('dateTime')
                condition_rule = tod_condition.attrib.get('rule')
                condition_operator = OpenScenarioParser.operators[condition_rule]
                atomic = TimeOfDayComparison(condition_date, condition_operator, condition_name)
            elif value_condition.find('StoryboardElementStateCondition') is not None:
                state_condition = value_condition.find('StoryboardElementStateCondition')
                element_name = state_condition.attrib.get('storyboardElementRef')
                element_type = state_condition.attrib.get('storyboardElementType')
                state = state_condition.attrib.get('state')
                if state == "startTransition":
                    atomic = OSCStartEndCondition(element_type, element_name, rule="START", name=state + "Condition")
                elif state == "stopTransition" or state == "endTransition" or state == "completeState":
                    atomic = OSCStartEndCondition(element_type, element_name, rule="END", name=state + "Condition")
                else:
                    raise NotImplementedError(
                        "Only start, stop, endTransitions and completeState are currently supported")
            elif value_condition.find('UserDefinedValueCondition') is not None:
                raise NotImplementedError("ByValue UserDefinedValue conditions are not yet supported")
            elif value_condition.find('TrafficSignalCondition') is not None:
                tl_condition = value_condition.find('TrafficSignalCondition')

                name_condition = tl_condition.attrib.get('name')
                traffic_light = OpenScenarioParser.get_traffic_light_from_osc_name(name_condition)

                tl_state = tl_condition.attrib.get('state').upper()
                if tl_state not in OpenScenarioParser.tl_states:
                    raise KeyError("CARLA only supports Green, Red, Yellow or Off")
                state_condition = OpenScenarioParser.tl_states[tl_state]

                atomic = WaitForTrafficLightState(
                    traffic_light, state_condition, name=condition_name)
            elif value_condition.find('TrafficSignalControllerCondition') is not None:
                raise NotImplementedError("ByValue TrafficSignalController conditions are not yet supported")
            else:
                raise AttributeError("Unknown ByValue condition")

        else:
            raise AttributeError("Unknown condition")

        if delay_atomic is not None and atomic is not None:
            new_atomic = py_trees.composites.Sequence("delayed sequence")
            new_atomic.add_child(delay_atomic)
            new_atomic.add_child(atomic)
        else:
            new_atomic = atomic

        return new_atomic

    @staticmethod
    def convert_maneuver_to_atomic(action, actor, catalogs):
        """
        Convert an OpenSCENARIO maneuver action into a Behavior atomic

        Note not all OpenSCENARIO actions are currently supported
        """
        maneuver_name = action.attrib.get('name', 'unknown')

        if action.find('GlobalAction') is not None:
            global_action = action.find('GlobalAction')
            if global_action.find('InfrastructureAction') is not None:
                infrastructure_action = global_action.find('InfrastructureAction').find('TrafficSignalAction')
                if infrastructure_action.find('TrafficSignalStateAction') is not None:
                    traffic_light_action = infrastructure_action.find('TrafficSignalStateAction')

                    name_condition = traffic_light_action.attrib.get('name')
                    traffic_light = OpenScenarioParser.get_traffic_light_from_osc_name(name_condition)

                    tl_state = traffic_light_action.attrib.get('state').upper()
                    if tl_state not in OpenScenarioParser.tl_states:
                        raise KeyError("CARLA only supports Green, Red, Yellow or Off")
                    traffic_light_state = OpenScenarioParser.tl_states[tl_state]

                    atomic = TrafficLightStateSetter(
                        traffic_light, traffic_light_state, name=maneuver_name + "_" + str(traffic_light.id))
                else:
                    raise NotImplementedError("TrafficLights can only be influenced via TrafficSignalStateAction")
            elif global_action.find('EnvironmentAction') is not None:
                weather_behavior = ChangeWeather(
                    OpenScenarioParser.get_weather_from_env_action(global_action, catalogs))
                friction_behavior = ChangeRoadFriction(
                    OpenScenarioParser.get_friction_from_env_action(global_action, catalogs))

                env_behavior = py_trees.composites.Parallel(
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name=maneuver_name)

                env_behavior.add_child(
                    oneshot_behavior(variable_name=maneuver_name + ">WeatherUpdate", behaviour=weather_behavior))
                env_behavior.add_child(
                    oneshot_behavior(variable_name=maneuver_name + ">FrictionUpdate", behaviour=friction_behavior))

                return env_behavior

            else:
                raise NotImplementedError("Global actions are not yet supported")
        elif action.find('UserDefinedAction') is not None:
            user_defined_action = action.find('UserDefinedAction')
            if user_defined_action.find('CustomCommandAction') is not None:
                command = user_defined_action.find('CustomCommandAction').attrib.get('type')
                atomic = RunScript(command, base_path=OpenScenarioParser.osc_filepath, name=maneuver_name)
        elif action.find('PrivateAction') is not None:
            private_action = action.find('PrivateAction')

            if private_action.find('LongitudinalAction') is not None:
                private_action = private_action.find('LongitudinalAction')

                if private_action.find('SpeedAction') is not None:
                    long_maneuver = private_action.find('SpeedAction')

                    # duration and distance
                    distance = float('inf')
                    duration = float('inf')
                    dimension = long_maneuver.find("SpeedActionDynamics").attrib.get('dynamicsDimension')
                    if dimension == "distance":
                        distance = float(long_maneuver.find("SpeedActionDynamics").attrib.get('value', float("inf")))
                    else:
                        duration = float(long_maneuver.find("SpeedActionDynamics").attrib.get('value', float("inf")))

                    # absolute velocity with given target speed
                    if long_maneuver.find("SpeedActionTarget").find("AbsoluteTargetSpeed") is not None:
                        target_speed = float(long_maneuver.find("SpeedActionTarget").find(
                            "AbsoluteTargetSpeed").attrib.get('value', 0))
                        atomic = ChangeActorTargetSpeed(
                            actor, target_speed, distance=distance, duration=duration, name=maneuver_name)

                    # relative velocity to given actor
                    if long_maneuver.find("SpeedActionTarget").find("RelativeTargetSpeed") is not None:
                        relative_speed = long_maneuver.find("SpeedActionTarget").find("RelativeTargetSpeed")
                        obj = relative_speed.attrib.get('entityRef')
                        value = float(relative_speed.attrib.get('value', 0))
                        value_type = relative_speed.attrib.get('speedTargetValueType')
                        continuous = relative_speed.attrib.get('continuous')

                        for traffic_actor in CarlaDataProvider.get_world().get_actors():
                            if 'role_name' in traffic_actor.attributes and traffic_actor.attributes['role_name'] == obj:
                                obj_actor = traffic_actor

                        atomic = ChangeActorTargetSpeed(actor,
                                                        target_speed=0.0,
                                                        relative_actor=obj_actor,
                                                        value=value,
                                                        value_type=value_type,
                                                        continuous=continuous,
                                                        distance=distance,
                                                        duration=duration,
                                                        name=maneuver_name)

                elif private_action.find('LongitudinalDistanceAction') is not None:
                    raise NotImplementedError("Longitudinal distance actions are not yet supported")
                else:
                    raise AttributeError("Unknown longitudinal action")
            elif private_action.find('LateralAction') is not None:
                private_action = private_action.find('LateralAction')
                if private_action.find('LaneChangeAction') is not None:
                    # Note: LaneChangeActions are currently only supported for RelativeTargetLane
                    #       with +1 or -1 referring to the action actor
                    lat_maneuver = private_action.find('LaneChangeAction')
                    target_lane_rel = float(lat_maneuver.find("LaneChangeTarget").find(
                        "RelativeTargetLane").attrib.get('value', 0))
                    # duration and distance
                    distance = float('inf')
                    duration = float('inf')
                    dimension = lat_maneuver.find("LaneChangeActionDynamics").attrib.get('dynamicsDimension')
                    if dimension == "distance":
                        distance = float(
                            lat_maneuver.find("LaneChangeActionDynamics").attrib.get('value', float("inf")))
                    else:
                        duration = float(
                            lat_maneuver.find("LaneChangeActionDynamics").attrib.get('value', float("inf")))
                    atomic = ChangeActorLateralMotion(actor,
                                                      direction="left" if target_lane_rel < 0 else "right",
                                                      distance_lane_change=distance,
                                                      distance_other_lane=1000,
                                                      name=maneuver_name)
                else:
                    raise AttributeError("Unknown lateral action")
            elif private_action.find('VisibilityAction') is not None:
                raise NotImplementedError("Visibility actions are not yet supported")
            elif private_action.find('SynchronizeAction') is not None:
                raise NotImplementedError("Synchronization actions are not yet supported")
            elif private_action.find('ActivateControllerAction') is not None:
                private_action = private_action.find('ActivateControllerAction')
                activate = strtobool(private_action.attrib.get('longitudinal'))
                atomic = ChangeAutoPilot(actor, activate, name=maneuver_name)
            elif private_action.find('ControllerAction') is not None:
                controller_action = private_action.find('ControllerAction')
                module, args = OpenScenarioParser.get_controller(controller_action, catalogs)
                atomic = ChangeActorControl(actor, control_py_module=module, args=args)
            elif private_action.find('TeleportAction') is not None:
                position = private_action.find('TeleportAction')
                atomic = ActorTransformSetterToOSCPosition(actor, position, name=maneuver_name)
            elif private_action.find('RoutingAction') is not None:
                private_action = private_action.find('RoutingAction')
                if private_action.find('AssignRouteAction') is not None:
                    # @TODO: How to handle relative positions here? This might chance at runtime?!
                    route_action = private_action.find('AssignRouteAction')
                    waypoints = OpenScenarioParser.get_route(route_action, catalogs)
                    atomic = ChangeActorWaypoints(actor, waypoints=waypoints, name=maneuver_name)
                elif private_action.find('FollowTrajectoryAction') is not None:
                    raise NotImplementedError("Private FollowTrajectory actions are not yet supported")
                elif private_action.find('AcquirePositionAction') is not None:
                    route_action = private_action.find('AcquirePositionAction')
                    osc_position = route_action.find('Position')
                    position = OpenScenarioParser.convert_position_to_transform(osc_position)
                    atomic = ChangeActorWaypointsToReachPosition(actor, position=position, name=maneuver_name)
                else:
                    raise AttributeError("Unknown private routing action")
            else:
                raise AttributeError("Unknown private action")

        else:
            if list(action):
                raise AttributeError("Unknown action: {}".format(maneuver_name))
            else:
                return Idle(duration=0, name=maneuver_name)

        return atomic
