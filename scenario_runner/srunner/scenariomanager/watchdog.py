#!/usr/bin/env python

# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a simple watchdog timer to detect timeouts
It is for example used in the ScenarioManager
"""
from __future__ import print_function

from threading import Timer
try:
    import thread
except ImportError:
    import _thread as thread


class Watchdog(object):

    """
    Simple watchdog timer to detect timeouts

    Args:
        timeout (float): Timeout value of the watchdog [seconds].
            If it is not reset before exceeding this value, a KayboardInterrupt is raised.

    Attributes:
        _timeout (float): Timeout value of the watchdog [seconds].
        _failed (bool):   True if watchdog exception occured, false otherwise
    """

    def __init__(self, timeout=1.0):
        """
        Class constructor
        """
        self._timeout = timeout + 1.0  # Let's add one second here to avoid overlap with other CARLA timeouts
        self._failed = False
        self._timer = None

    def start(self):
        """
        Start the watchdog
        """
        self._timer = Timer(self._timeout, self._event)
        self._timer.daemon = True
        self._timer.start()

    def update(self):
        """
        Reset watchdog.
        """
        self.stop()
        self.start()

    def _event(self):
        """
        This method is called when the timer triggers. A KayboardInterrupt
        is generated on the main thread and the watchdog is stopped.
        """
        print('Watchdog exception - Timeout of {} seconds occured'.format(self._timeout))
        self._failed = True
        self.stop()
        thread.interrupt_main()

    def stop(self):
        """
        Stops the watchdog.
        """
        if self._timer is not None:
            self._timer.cancel()

    def get_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return not self._failed
