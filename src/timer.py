import time


class Timer():
    def __init__(self):
        self.start()

    def start(self):
        self.start_time = time.time()
        self.ptime = self.start_time
        self.lap_list = []
        self.split_list = []

    def lap(self):
        t = time.time()
        lap = t - self.ptime
        self.ptime = t
        self.lap_list.append(lap)
        return lap

    def split(self):
        t = time.time()
        sp = t - self.start_time
        self.split_list.append(sp)
        return sp

    @property
    def laps(self):
        return self.lap_list

    @property
    def splits(self):
        return self.split_list
