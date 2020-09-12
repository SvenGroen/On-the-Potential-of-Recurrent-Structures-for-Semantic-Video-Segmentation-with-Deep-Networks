import time
import sys


class TimeLogger:
    def __init__(self, restart_time):
        self.instantiated = time.time()
        self.restart = restart_time
        self.avrg_batch_time = None  # 5 minutes
        self.batch_start = 0
        self.last_update = self.instantiated

    def check_for_restart(self):
        self.update()
        self.batch_start = time.time()
        return self.batch_start + self.avrg_batch_time - self.instantiated > self.restart

    def update(self):
        passed_time = time.time() - self.last_update
        self.avrg_batch_time = (self.avrg_batch_time + passed_time) / 2 if (
                self.avrg_batch_time is not None) else passed_time
        self.last_update = time.time()

    def get_status(self):
            return f"\nPassed time = {time.time() - self.instantiated} and restart time: {self.restart} and average_batch_time = {self.avrg_batch_time}\n"

# sys.stderr.write(f"\nBatch start= {self.batch_start} and instantiated: {self.instantiated} 
# and bs_start + avrg -instant = {self.batch_start + self.avrg_batch_time - self.instantiated } > restart: {self.restart}\n")
