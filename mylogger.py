# logs all the sys.stdout to a given file
# source: https://stackoverflow.com/a/24583265

import sys, os

class MyLogger(object):
    "Lumberjack class - duplicates sys.stdout to a log file and it's okay"
    #source: https://stackoverflow.com/q/616645
    def __init__(self, filename="logout.txt", mode="a", buff=0):
        head,_ = os.path.split(filename)
        if len(head)>0 and not os.path.exists(head): os.makedirs(head)
        self.stdout = sys.stdout
        self.file = open(filename, mode, buff)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.file.write(message)
        return self.stdout.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None