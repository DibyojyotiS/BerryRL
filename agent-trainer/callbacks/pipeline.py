from typing import Any, Callable, Dict, List
from queue import Queue
from threading import Thread


class ThreadSafePrinter:
    def __init__(self) -> None:
        self.queue = Queue()
        self.daemonThread = Thread(
            target=self.printer, args=(self.queue, ), daemon=True
        )
        self.daemonThread.start()
        self.started = True

    def __call__(self, message) -> Any:
        if not self.started: self.start()
        self.queue.put(message)

    def printer(self, queue:Queue):
        while True:
            message = queue.get()
            if message == "end": break
            print(message)

    def __del__(self):
        self.queue.put("end")

    def __delete__(self, instance):
        if instance.started:
            instance.queue.put("end")


class DaemonPipe:
    def __init__(self, pipeline:List[Callable]=[]) -> None:
        self.pipeline = pipeline
        self.started = False

    def append_stage(self, stage_callable):
        self.pipeline.append(stage_callable)

    def start(self):
        self.queue = Queue()
        self.daemonThread = Thread(
            target=self.consumer, args=(self.queue, ), daemon=True
        )
        self.daemonThread.start()
        self.started = True

    def consumer(self, queue:Queue):
        while True:
            object = queue.get()
            if object == "end": break
            self._execute_pipe(object)

    def stop(self):
        self.queue.put("end")

    def _execute_pipe(self, info:Dict[str,Any]):
        for stage in self.pipeline:
            info = stage(info)

    def __call__(self, object) -> None:
        if not self.started: self.start()
        self.queue.put(object)

    def __del__(self):
        self.queue.put("end")

    def __delete__(self, instance):
        if instance.started:
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

    bg = DaemonPipe([cp])
    bg.start()
    bg({'x':10, 'y':20})
