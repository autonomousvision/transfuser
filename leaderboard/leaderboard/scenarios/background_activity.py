#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario


BACKGROUND_ACTIVITY_SCENARIOS = ["BackgroundActivity"]


class BackgroundActivity(BasicScenario):

    """
    Implementation of a scenario to spawn a set of background actors,
    and to remove traffic jams in background traffic

    This is a single ego vehicle scenario
    """

    category = "BackgroundActivity"

    town_amount = {
        'Town01': 120,
        'Town02': 100,
        'Town03': 120,
        'Town04': 200,
        'Town05': 120,
        'Town06': 150,
        'Town07': 110,
        'Town08': 180,
        'Town09': 300,
        'Town10': 120,
    }

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, timeout=35 * 60):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.debug = debug_mode

        self.timeout = timeout  # Timeout of scenario in seconds

        super(BackgroundActivity, self).__init__("BackgroundActivity",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=True,
                                                 criteria_enable=True)

    def _initialize_actors(self, config):

        town_name = config.town
        if town_name in self.town_amount:
            amount = self.town_amount[town_name]
        else:
            amount = 0

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background')

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        pass

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        pass

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
