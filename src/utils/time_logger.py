import time


class TimeLogger:
    def __init__(self, restart_time):
        self.instantiated = time.time()
        self.restart = restart_time
        self.avrg_batch_time = 60 * 5  # 5 minutes
        self.batch_start = 0

    def check_for_restart(self):
        self.update()
        self.batch_start = time.time()
        return self.batch_start + self.avrg_batch_time - self.instantiated > self.restart

    def update(self):
        passed_time = time.time() - self.instantiated
        self.avrg_batch_time = (self.avrg_batch_time + passed_time) / 2
