## Table of Contents
* [Latest Changes](#latest-changes)
* [CARLA ScenarioRunner 0.9.10](#carla-scenariorunner-0910)
* [CARLA ScenarioRunner 0.9.9](#carla-scenariorunner-099)
* [CARLA ScenarioRunner 0.9.8](#carla-scenariorunner-098)
* [CARLA ScenarioRunner 0.9.7](#carla-scenariorunner-097)
* [CARLA ScenarioRunner 0.9.6](#carla-scenariorunner-096)
* [CARLA ScenarioRunner 0.9.5.1](#carla-scenariorunner-0951)
* [CARLA ScenarioRunner 0.9.5](#carla-scenariorunner-095)
* [CARLA ScenarioRunner 0.9.2](#carla-scenariorunner-092)

## Latest Changes

## CARLA ScenarioRunner 0.9.10
### :rocket: New Features
* Renamed some agent labels inside Jenkins CI pipelines for new standard proposals.
* Added support for Jenkins CI pipelines doing automated testing and docker images creation.
* **Very important:** CarlaActorPool has been removed and all its functions moved to the CarlaDataProvider:
    - The spawning functions have been refactored. All the *setup* functions have been removed, and its functionalities moved to their *request* counterparts. For example, previously *request_new_actor* just called *setup_actor*, but now *setup_actor* no longer exists, and the spawning is done via *request_new_actor*. They have also been unified and are now more consistent.
    - Changed *ActorConfiguration* to *ActorConfigurationData.parse_from_node*
    - Renamed the _map_ element at routes to _town_, matching the scenario configuration files

* Added new environment variables needed. They can be seen at (Docs/getting_scenariorunner.md).
* Improved the visual display of the information from the *output* and *file* arguments.
* Routes are now deterministic in regards to the spawning scenarios when more than one are at the same location
* The BackgroundActivity functionality has been unchanged but some tweaks have been made, fixing a previous patch. As a result, the *amount* parameter at *ActorConfigurationData* has been removed.
* Remade how ScenarioRunner reads the scenarios files. It now reads all scenarios inside the *srunner/scenarios* folder without needing to import them. Scenarios outside that folder will still need the *--additionalScenario* argument.
* The new weather parameters (related to fog) are now correctly read when running scenarios outside routes.
* Enable weather animation during scenario execution (requires ephem pip package)
* Changed manual control to be in par with the CARLA version. Among others, added vehicle lights, recording and some new sensors
* Removed unsupported scenarios (ChallengeBasic and BackgroundActivity, VehicleTurnLeftAtJunction)
* Added a new metrics module, which gives access to all the information about a scenario in order to allow the user to extract any desired information about the simulation. More information [here](metrics_module.md)
* Removed the default randomness at the ControlLoss scenario
* OpenSCENARIO support:
    - Added support for controllers and provided default implementations for vehicles and pedestrians. This required changing the handling of actors, which results in that now all actors are controlled by an OSC controller. Supported controllers:
        - Pedestrian controller
        - NPC vehicle controller (based on CARLA LocalPlanner)
        - Simple vehicle controller to set velocities not brake/throttle, and consider obstacles in the forward-facing region.
        - External controller (to forward control to external entities)
    - Added initial speed support for pedestrians for OpenSCENARIO
    - Support for EnvironmentActions within Story (before only within Init). This allows changing weather conditions during scenario execution
    - Added support for RelativeSpeedCondition
    - Added support for AccelerationCondition
    - Added support for TimeOfDayCondition
    - Added support for OffroadCondition
    - Added support for CollisionCondition
    - Added support for EndOfRoadCondition
    - Added support for TimeHeadwayCondition
    - Added support for TrafficSignalCondition
    - Added support for AcquirePositionAction
    - Extended FollowLeadingVehicle example to illustrate weather changes
    - Created example scenarios to illustrate usage of controllers and weather changes
    - Reworked the handling of Catalogs to make it compliant to the 1.0 version (relative paths have to be relative to the scenario file)
    - The RoadNetwork can be defined as global Parameter
    - Fixed handling of relative positions with negative offset
    - Added support for local ParamaterDeclarations
    - Added support for Parameters within Catalogs
    - Added support for ParameterAssignments for CatalogReferences
    - Fixed name handling of Parameters: Parameter declerations must not start with a leading '$', but when the parameter is used a leading '$' is required.
    - Fixed use of Parameters for multiple instances of the same Catalog element
    - Fixed use of relative initial positions for any actor
    - Added possibility to use synchronous execution mode with OpenSCENARIO
    - Fixed use of relative paths in CustomCommandAction
    - Fixed use of ControllerCatalogs
* Atomics:
    - Several new atomics to enable usage of OSC controllers
    - WeatherBehavior to simulate weather over time
    - UpdateWeather to update weather to a new setting, e.g. sun to rain
    - UpdateRoadFriction to update the road friction while running
    - new RelativeVelocityToOtherActor trigger condition, used to compare velocities of two actors
    - new TriggerAcceleration trigger condition which compares a reference acceleration with the actor's one.
    - new TimeOfDayComparison trigger condition, comparing the simulation time (set up by the new weather system) with a given *datetime*.
    - Added new *OffRoadTest* criteria.
    - Added new *EndofRoadTest* criteria, to detect when a vehicle changes between OpenDRIVE roads.
    - CollisionTest criterion can now filter the collisions for a specific actor, or actor type_id.
    - Added a *duration* argument to *OnSidewalkTest* criteria, which makes the criteria fail after a certain time has passed, instead of doing so immediately. The default behavior has been unchanged.
    - InTimeToArrivalToVehicle has had its two actor arguments swapped, to match all the other behaviors.
    - Added *along_route* flag to InTimeToArrivalToVehicle, to take into account the topology of the road
    - Changed the inputs to TrafficLightStateSetter to match the other atomics, but the functionality remains unchanged

### :bug: Bug Fixes
* Fixed bug causing parsing RelativeTargetSpeed tag to fail. 
* Fixed missing 'six' in requirements.txt
* Support OpenSCENARIO parameters also if they're only part of a string value
* Support Routes in Catalogs
* Fix parsing of properties within ControllerCatalogs
* Add cleanup of instantiated OpenSCENARIO controllers
* Do not register SIGHUP signal in windows
* Fixed initial speed of vehicles using OpenSCENARIO
* Fixed bug causing an exception when calling BasicScenario's *_initialize_actors* with no other_actors.
* Fixed bug causing the route to be downsampled (introduced by mistake at 0.9.9)
* Fixed bug causing _output_ argument to not display the correct number with _InRouteTest_ and _RouteCompletionTest_ criterias (the succces and failure was correctly displayed)
* Fixed bug causing OpenSCENARIO's SpeedCondition to not work as intended
* Fixed bug causing CollisionConditions not to work properly in OpenSCENARIO
* Fixed bug causing the *group:* functionality to behave incorrectly when moving scenarios around.
* Fixed bug causing FollowLeadingVehicle and FollowLeadingVehicleWithObstacle scenarios to not properly end
* Fixed bug causing CollisionTest to ignore multiple collisions with scene objects
* Fixed bug causing NoSignalJunctionCrossing to not output the results of the scenario
* Fixed bug causing SyncArrival to fail when the actor was destroyed after the behavior ended
* Fixed bug with ending roads near stop signals to break the simulation
* Fixed exception bug in spawn function of CarlaDataProvider
* Fixed access to private member of CARLA LocalPlanner inside OSC NpcVehicleControl
* Fixed handling of OSC LanePosition (#625)
* Fixed bug causing the route repetitions to spawn different background activity
* Fixed bug causing the rotate_point function inside RunningRedLightTest to not function properly.
### :ghost: Maintenance
* Exposed traffic manager port flag to enable the execution of multiple scenarios on a single machine.

## CARLA ScenarioRunner 0.9.9
### :rocket: New Features
* OpenSCENARIO support:
    - Support for OpenSCENARIO 1.0 (a converter for old scenarios is available)
    - Added support for position with Lane information (roadId and laneId)
    - Added support to use a non-CARLA OpenDRIVE map (instead of CARLA towns)
    - Added support for TimeOfDay tag
    - Added support for scenarios with no actors
* Scenario updates:
    - Scenarios that are part of RouteScenario have had their triggering condition modified. This will only activate when a certain parameter is set, and if not, the old trigger condition will still be applied.
* Atomics:
    - ChangeAutopilot now calls a TM instance, and allows to change its parameters
    - Added WaitUntilInFront behavior and InTimeToArrivalToVehicleSideLane trigger condition, useful for cut ins
    - Added new trigger condition, AtRightestLane, which checks if the actor is at the rightmost driving lane
    - Added new criteria, ActorSpeedAboveThresholdTest, useful to check if the ego vehicle has been standing still for long periods of time.
* Setting up actors in batch now also randomizes their colors
* When running routes, the weather parameters of each route can now be changed at will. Check the first route at srunner/data/routes_training.xml to see the correct format to do so. By default the weather is now a sunny midday.
* **Important** All challenge related content has been removed. This functionality has been improved and is now part of the [Leaderboard](https://github.com/carla-simulator/leaderboard). As a consequence:
    - The path to the autoagents has changed from .../challenge/autoagents to .../autoagents
    - The path to the route and scenario descriptions has changed from .../challenge to .../data
### :bug: Bug Fixes
* Fixed spawning bugs for scenario DynamicObjectCrossing when it is part of a route
* Fixed spawning bugs for scenarios VehicleTurningRight, VehicleTurningLeft when they are part of a route
* Fixed bug causing the GPS coordinates given to the agents to be wrongly calculated
* Fixed bug when setting up actors in batch causing to ignore the spawn points given.
* Fixed bug where CollisionTest was counting as multiple hits collisions that displaced the actor for a long distance.
* Fixed bug causing the simulation to end after running in synchronous mode
* Fixed bug when using the WaypointFollower atomic to create new LocalPlanners for on-the-fly created actors (#502)
* Fixed bug causing the scenarios to run faster than real time.
### :ghost: Maintenance
* Removed perform_carla_tick() function at CarlaDataProvider, which was a workaround for world.tick()


## CARLA ScenarioRunner 0.9.8
### :rocket: New Features
* Added "--timeout" command line parameter to set a user-defined timeout value
* Scenario updates:
    - Changed traffic light behavior of scenarios 7, 8 and 9. The new sequence is meant to greatly improve the chances of the ego vehicle having to interact at junctions.
* OpenSCENARIO support:
    - Added initial support for Catalogs (Vehicle, Pedestrian, Environment, Maneuver, and and MiscObject types only)
### :bug: Bug Fixes
* Fixed #471: Handling of weather parameter (cloudyness -> cloudiness adaption)
* Fixed #472: Spawning issue of pedestrians in OpenSCENARIO
* Fixed #374: Usage of evaluation critieria with multiple ego vehicles in OpenSCENARIO
* Fixed #459: Add initial support for Catalogs (Vehicle, Pedestrian, Environment, Maneuver, and and MiscObject types only)
* Fixed wrong StandStill behavior which return SUCCESS immediatly on a standing actor
* Fixed scenario bug causing junction related scenarios (4, 7, 8 and 9) to not spawn due to lane changes.
### :ghost: Maintenance
* Added watchdog to ScenarioManager to handle timeouts and CARLA crashes
* Added timeout for CARLA tick() calls to avoid blocking CARLA server calls


## CARLA ScenarioRunner 0.9.7
**This is the _first_ release to work with CARLA 0.9.7 (not the patch versions 0.9.7.x)**
### :rocket: New Features
* Challenge routes can be directly executed with the ScenarioRunner using the --route option
* Agents can be used with the ScenarioRunner (currently only for route-based scenarios)
* New scenarios:
    - Added example scenario for lane change
    - Added cut-in example scenario
* Scenario updates:
    - Scenarios 7 to 10 are now visible when running routes (instead of being triggered in the background). Their
      methodology has remained unchanged
* Scenario atomics:
    - Added new OutsideRouteLanesTest atomic criter that encompasses both SidewalkTest and WrongLaneTest, returning
      the percentage of route that has been traveled outside the lane.
    - InRouteTest is now more forgiving. The max distance has been increased, but staying above the previous one will eventually 
      also cause failure
    - Changed SidewalkTest atomic criteria to also track other type of out of lane conditions
    - SidewalkTest and WrongLaneTest atomic criterias now track the amount of meters traversed
    - CollisionTest atomic criteria now correctly ignores multiple micro-collisions with the same object
    - Added LaneChange and TrafficLightSateSetter behavior atomics
    - Added AccelerateToCatchUp behavior atomic
    - Added get_transform() method for CarlaDataProvider
    - Added support for weather conditions
    - Added basic version check to ensure usage of correct CARLA version
    - WaypointFollower atomic can handle pedestrians
    - Extensions in WaypointFollower atomic for consecutive WaypointFollowers (one WF cancels the previous one)
* Extended OpenScenario support:
    - Added support for UserDefinedActions (e.g. to run additional scripts)
    - Added init speed behavior for vehicles
    - Added support for relative velocities
    - Extended convert_position_to_transform with RelativeWorld, RelativeObject and RelativeLane osc_positions
    - Added new trigger atomics InTriggerDistanceToOSCPosition and InTimeToArrivalToOSCPosition to support relative osc_positions
    - Added new atomic behaviour ActorTransformSetterToOSCPosition
    - Workaround for relative osc_positions: World is started earlier to support relative osc_positions in story init
    - Added delay condition support in convert_condition_to_atomic
    - Added support for pedestrians
    - Full support for SimulationTime condition
    - Added weather support
    - Updated implementation to be closer to upcoming OpenSCENARIO standard
    - AfterTermination, AtStart conditions are supported
    - Added initial support for lateral action: LaneChange
    - Added initial support for OSCGlobalAction to set state of traffic signal
    - FollowRoute action is supported for vehicles and pedestrians, for global world positions.
    - Added support for RoadCondition: Friction
    - Redundant rolename object property is no longer required
    - Added support for global parameters
    - Fixed coordinate system to use right-hand as default. Left-hand CARLA system can be used by adding "CARLA:" at the start of the description in the FileHeader.
    - Added support to change actor color
    - Added support for a default actor model, in case the stated model is not available
    - Added support for MiscObjects (besides vehicles and pedestrians)
    - Reworked traffic signal handling: The name has to start now either with "id=" or "pos=" depending on whether the position or id is used as unique identifier
    - Actor physics can now be set via Object Properties (<Property name="physics" value="off" />)
### :bug: Bug Fixes
* Fixed wrong handling of OpenSCENARIO ConditionGroups, which should be handled as parallel composites, not sequences
* Fixed #443: Repetitions in OpenSCENARIO were not properly working
* Fixed bug causing RunningStopTest atomic criteria to trigger when lane changing near a STOP signal
* Fixed bug causing RunningRedLightTest atomic criteria to occasionally not trigger
* Fixed bug causing occasional frame_errors
* Fixed #426: Avoid underground vehicles fall forever by disabling physics when spawning underground.
* Fixed #427: Removed unnecessary warnings when using get_next_traffic_light() with non-cached locations
* Fixed missing ego_vehicle: compare actor IDs instead of object in CarlaDataProvider in get_velocity, get_transform and get_location
* Avoided use of 'controller.ai.walker' as walker type in DynamicObjectCrossing scenario
* Fixed WaypointFollower behavior to use m/s instead of km/h
* Fixed starting position of VehicleTurnLeft/Right scenarios
* Fixed spawn_point modification inside CarlaActorPool.setup_actor()
* Fixed result of DrivenDistanceTest
* Fixed exception in manual_control on fps visualization
* Cleanup of pylint errors for all autonomous agents
* Fixed randomness of route-based scenarios
* Fixed usage of radians instead of degrees for OpenSCENARIO
* Fixed ActorTransformSetter behavior to avoid vehicles not reaching the desired transform
* Fixed spawning of debris for ControlLoss scenario (Scenario01)
* Fixed CTRL+C termination of ScenarioRunner
### :ghost: Maintenance
* Increased speed of actor initialization by using CARLA batch mode and buffering CARLA blueprint library
* Split of behaviors into behaviors and conditions
* Moved atomics into new submodule scenarioatomics
* Updated documentation for all behaviors, conditions and test criteria
* Refactoring of scenario configurations and parsers
* Extended WaypointFollower atomic behavior to be able to use the current actor speed
* Removed usage of 'import *' to have cleaner Python imports
* Removed broad-except and bare-except where possible
* Python-Scenarios: Removed obsolete categories
* ScenarioRunner: Removed scenario dictonary, use imports directly
* CarlaDataProvider: Simplified update_light_states() to remove code duplication
* Timer: class TimeOut() is derived from SimulationTimeCondition() to  avoid code duplication
* Moved backported py_trees classes and methods to tools/py_trees_port.py to avoid code duplication
* Removed setup_environment.sh
* Adaptions to CARLA API Changes
     - Renamed GnssEvent to GnssMeasurement

## CARLA ScenarioRunner 0.9.6
**This is the _first_ release to work with CARLA 0.9.6**
### :ghost: Maintenance
* Adapted to CARLA API changes
    - Frame rate is set now via Python
    - Renamed frame_count and frame_number to frame
    - Removed wait_for_tick() calls


## CARLA ScenarioRunner 0.9.5.1
**This is the _last_ release that works with CARLA 0.9.5**
### :rocket: New Features
* Added initial support for OpenScenario v0.9.1
* Added support for multiple ego vehicles plus an example
* Added commandline option for output directory
* Added option to load external scenario implementations (in python)
* Added option to scenario_runner to load external scenario XMLs
* Atomic behaviors:
    - Extended KeepVelocity atomic behavior to support duration/distance
      based termination
    - Extended StandStill atomic behavior to support duration based
      termination
    - Added behavior to activate/deactivate autopilot
### :bug: Bug Fixes
* Fixed WaypointFollower initialization


## CARLA ScenarioRunner 0.9.5
**This is the _first_ release to work with CARLA 0.9.5**
### :rocket: New Features
* Added support for CARLA challenge
    - Added logging functionalities to challenge_evaluator_routes.py
    - Added wall clock timeout for the CARLA challenge
    - Added background scenario to generate dynamic traffic using autopilot
    - Updated compatibility with Python 2.7 for the challenge evaluator
    - Updated WaypointFollower behavior
    - Added detect_lane_obstacle() helper function which identifies if an obstacle is present in front of the reference actor
    - Added test to detect vehicles running a stop
    - Updated the reference position for a scenario is now called trigger_point
    - Added universal access to the map without re-calling get_map()
    - Added criteria_enable flag to enable/disable criteria tree
    - Added multiple helper methods for generic scenario execution.
    - Added pseudo-sensors for SceneLayoutMeasurements and ObjectMeasurements for Track4 of the CARLA AD challenge
    - Added track identification for autonomous_agent.py
    - Added HDMap pseudo-sensor
    - Added new traffic event logger
    - Added various helper methods to allow generic scenario execution
    - Added method to calculate distance along a route
    - In challenge mode spawn exception are caught and the corresponding scenario is removed
* Added new atomic behaviors using py_trees behavior tree library
    - BasicAgentBehavior: drive to target location using CARLA's BasicAgent
    - StandStill: check if a vehicle stands still
    - InTriggerDistanceToNextIntersection: check if a vehicle is within certain distance with respect to the next intersection
    - WaypointFollower: follows auto-generated waypoints indefinitely or follows a given waypoint list
    - HandBrakeVehicle: sets the handbrake value for a given actor
    - ActorDestroy: destroys a given actor
    - ActorTransformSetter: sets transform of given actor
    - ActorSource: creates actors indefinitely around a location if no other vehicles are present within a threshold
    - ActorSink: indefinitely destroys vehicles that wander close to a location within a threshold
    - InTriggerDistanceToLocationAlongRoute: check if an actor is within a certain distance to a given location along a given route
* Added new atomic evaluation criteria
    - Added running red light test
    - Added running stop test
    - Added wrong way test
* Added NHTSA Traffic Scenarios
    - Updated all traffic scenarios to let the other actors appear upon scenario triggering and removal on scenario end
    - ManeuverOppositeDirection: hero vehicle must maneuver in the opposite lane to pass a leading vehicle.
    - OtherLeadingVehicle: hero vehicle must react to the deceleration of leading vehicle and change lane to avoid collision and follow the vehicle in changed lane
    - SignalizedJunctionRightTurn: hero vehicle must turn right into the same direction of another vehicle crossing straight initially from a lateral direction and avoid collision at a signalized intersection.
    - SignalizedJunctionLeftTurn : hero vehicle is turning left at signalized intersection, cuts across the path of another vehicle coming straight crossing from an opposite direction.
### :bug: Bug Fixes
* Fixed SteerVehicle atomic behavior to keep vehicle velocity    
### :ghost: Maintenance
* Reworked scenario execution
    - Updated folder structure and naming convention in lowercase
    - Extended CarlaDataProvider with method to get next relevant traffic light
    - Every scenario has to have a configuration provided as XML file.
      Currently there is one XML file for each scenario class
    - The scenario runner is now responsible for spawning/destroying the ego vehicle
    - Added a CarlaActorPool to share scenario-related actors between scenarios and the scenario_runner
    - Renamed vehicle -> actor
    - If all scenarios in one configurations file should be executed, the scenario_runner can be started with --scenario group:<CONFIG_FILE>
    - Generalized ControlLoss and FollowLeadingVehicle scenarios
    - Added randomization option to scenario_runner and scenarios
    - The scenario behavior always starts with a wait behavior until the ego vehicle reached the scenario starting position
    - Created method _initialize_actors in basic scenario that can be overridden for scenario specific actor initialization
* Updated NHTSA Traffic Scenarios
    - OppositeVehicleRunningRedLight: Updated to allow execution at different locations    


## CARLA ScenarioRunner 0.9.2
**This release is designed to work with CARLA 0.9.2**
### :rocket: New Features
* Added Traffic Scenarios engine to reproduce complex traffic situations for training and evaluating driving agents
* Added NHTSA Traffic Scenarios
    - FollowLeadingVehicle: hero vehicle must react to the deceleration of a leading vehicle
    - FollowLeadingVehicleWithObstacle: hero vehicle must react to a leading vehicle due to an obstacle blocking the road
    - StationaryObjectCrossing: hero vehicle must react to a cyclist or pedestrian blocking the road
    - DynamicObjectCrossing: hero vehicle must react to a cyclist or pedestrian suddenly crossing in front of it
    - OppositeVehicleRunningRedLight: hero vehicle must avoid a collision at an intersection regulated by traffic lights when the crossing traffic runs a red light
    - NoSignalJunctionCrossing: hero vehicle must cross a non-signalized intersection
    - VehicleTurningRight: hero vehicle must react to a cyclist or pedestrian crossing ahead after a right turn
    - VehicleTurningLeft: hero vehicle must react to a cyclist or pedestrian crossing ahead after a left turn
    - ControlLoss: Hero vehicle must react to a control loss and regain its control
* Added atomic behaviors using py_trees behavior trees library
    - InTriggerRegion: new behavior to check if an object is within a trigger region
    - InTriggerDistanceToVehicle: check if a vehicle is within certain distance with respect to a reference vehicle
    - InTriggerDistanceToLocation: check if a vehicle is within certain distance with respect to a reference location
    - TriggerVelocity: triggers if a velocity is met
    - InTimeToArrivalToLocation:  check if a vehicle arrives within a given time budget to a reference location
    - InTimeToArrivalToVehicle: check if a vehicle arrives within a given time budget to a reference vehicle
    - AccelerateToVelocity: accelerate until reaching requested velocity
    - KeepVelocity: keep constant velocity
    - DriveDistance: drive certain distance
    - UseAutoPilot: enable autopilot
    - StopVehicle: stop vehicle
    - WaitForTrafficLightState: wait for the traffic light to have a given state
    - SyncArrival: sync the arrival of two vehicles to a given target
    - AddNoiseToVehicle: Add noise to steer as well as throttle of the vehicle
