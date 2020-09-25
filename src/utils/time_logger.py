import time
import sys


class TimeLogger:
    """
    Time logger keeps track of how long the current script is running. If a set restart time is exceeded a signal
    will be returned such that a restart of the script can be done

    :param restart_time: the time that should not be exceeded
    """
    def __init__(self, restart_time):
        """
        See help(TimeLogger) for help
        """
        self.instantiated = time.time()
        self.restart = restart_time
        self.avrg_batch_time = None  # 5 minutes
        self.batch_start = 0
        self.last_update = self.instantiated

    def check_for_restart(self):
        """
        Checks if runtime exceeds restart time
        :return: true if restart time is exceeded, else false
        """
        self.update()
        self.batch_start = time.time()
        return self.batch_start + self.avrg_batch_time - self.instantiated > self.restart

    def update(self):
        """
        updates current timestamps.
        """
        passed_time = time.time() - self.last_update
        self.avrg_batch_time = (self.avrg_batch_time + passed_time) / 2 if (
                self.avrg_batch_time is not None) else passed_time
        self.last_update = time.time()

    def get_status(self):
        """
        Gives information on how much time already passed
        :return: information on passed time, restart time and average batch time
        """
        return f"\nPassed time = {time.time() - self.instantiated} and restart time: {self.restart} and average_batch_time = {self.avrg_batch_time}\n"

