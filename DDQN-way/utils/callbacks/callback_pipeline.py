from typing import Callable, List
from queue import Queue
from threading import Thread


class BackgroundThread:
    def __init__(self, callable:Callable) -> None:
        self.callable = callable

    def startBackGroundProcess(self):
        self.queue = Queue()
        self.daemonThread = Thread(target=self.consumer, args=(self.queue, ), daemon=True)
        self.daemonThread.start()

    def consumer(self, queue:Queue):
        while True:
            object = queue.get()
            if object == "end": break
            self.callable(object)

    def __call__(self, object) -> None:
        self.queue.put(object)

    def stop(self):
        self.queue.put("end")

    def __del__(self):
        self.queue.put("end")

    def __delete__(self, instance):
        instance.queue.put("end")


class CallbackPipeline:
    def __init__(self, pipeline:List[Callable]=[]) -> None:
        self.pipeline = pipeline

    def append_stage(self, stage_callable):
        self.pipeline.append(stage_callable)

    def __call__(self, info_dict:dict) -> None:
        for stage in self.pipeline:
            info_dict = stage(info_dict)


if __name__ == "__main__":

    def printX(obj):
        print(obj["x"])
        return obj

    cp = CallbackPipeline(pipeline=[
        printX,
        print
    ])

    bg = BackgroundThread(cp)
    bg.startBackGroundProcess()
    bg({'x':10, 'y':20})
