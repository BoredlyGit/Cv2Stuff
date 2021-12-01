import time
import sys


class ProgressBar:
    def __init__(self, end, width=10):
        self._progress = 0
        self.end = end
        self.width = width

    def __str__(self):
        hashes = '#' * int((self.progress / self.end) * self.width)
        return f"[{hashes}{'-' * (self.width - len(hashes))}] {self.progress}/{self.end}"

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, val):
        self._progress = val
        self.update_stdout()

    def update_stdout(self):
        print(f"\r{str(self)}", end="" if self._progress < self.end else "\n")

#
# p = ProgressBar(10, 20)
# for i in range(1, 11):
#     time.sleep(0.5)
#     p.progress = i