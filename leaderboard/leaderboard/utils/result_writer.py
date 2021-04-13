#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains the result gatherer and write for CARLA scenarios.
It shall be used from the ScenarioManager only.
"""

from __future__ import print_function

import time
from tabulate import tabulate


class ResultOutputProvider(object):

    """
    This module contains the _result gatherer and write for CARLA scenarios.
    It shall be used from the ScenarioManager only.
    """

    def __init__(self, data, global_result):
        """
        - data contains all scenario-related information
        - global_result is overall pass/fail info
        """
        self._data = data
        self._global_result = global_result

        self._start_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(self._data.start_system_time))
        self._end_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(self._data.end_system_time))

        print(self.create_output_text())

    def create_output_text(self):
        """
        Creates the output message
        """

        # Create the title
        output = "\n"
        output += "\033[1m========= Results of {} (repetition {}) ------ {} \033[1m=========\033[0m\n".format(
            self._data.scenario_tree.name, self._data.repetition_number, self._global_result)
        output += "\n"

        # Simulation part
        system_time = round(self._data.scenario_duration_system, 2)
        game_time = round(self._data.scenario_duration_game, 2)
        ratio = round(self._data.scenario_duration_game / self._data.scenario_duration_system, 3)

        list_statistics = [["Start Time", "{}".format(self._start_time)]]
        list_statistics.extend([["End Time", "{}".format(self._end_time)]])
        list_statistics.extend([["Duration (System Time)", "{}s".format(system_time)]])
        list_statistics.extend([["Duration (Game Time)", "{}s".format(game_time)]])
        list_statistics.extend([["Ratio (System Time / Game Time)", "{}".format(ratio)]])

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n\n"

        # Criteria part
        header = ['Criterion', 'Result', 'Value']
        list_statistics = [header]

        for criterion in self._data.scenario.get_criteria():

            actual_value = criterion.actual_value
            expected_value = criterion.expected_value_success
            name = criterion.name

            result = criterion.test_status

            if result == "SUCCESS":
                result = '\033[92m'+'SUCCESS'+'\033[0m'
            elif result == "FAILURE":
                result = '\033[91m'+'FAILURE'+'\033[0m'

            if name == "RouteCompletionTest":
                actual_value = str(actual_value) + " %"
            elif name == "OutsideRouteLanesTest":
                actual_value = str(actual_value) + " %"
            elif name == "CollisionTest":
                actual_value = str(actual_value) + " times"
            elif name == "RunningRedLightTest":
                actual_value = str(actual_value) + " times"
            elif name == "RunningStopTest":
                actual_value = str(actual_value) + " times"
            elif name == "InRouteTest":
                actual_value = ""
            elif name == "AgentBlockedTest":
                actual_value = ""

            list_statistics.extend([[name, result, actual_value]])

        # Timeout
        name = "Timeout"

        actual_value = self._data.scenario_duration_game
        expected_value = self._data.scenario.timeout

        if self._data.scenario_duration_game < self._data.scenario.timeout:
            result = '\033[92m'+'SUCCESS'+'\033[0m'
        else:
            result = '\033[91m'+'FAILURE'+'\033[0m'

        list_statistics.extend([[name, result, '']])

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n"

        return output
