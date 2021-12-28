import threading

class MyThread(threading.Thread):
    def __init__(self, condition):
        threading.Thread.__init__(self)
        self.condition = condition

    def run(self):
        print "%s done" % threading.current_thread()
        with self.condition:
            self.condition.notify()


condition = threading.Condition()
condition.acquire()

thread = MyThread(condition)
thread.start()

condition.wait()