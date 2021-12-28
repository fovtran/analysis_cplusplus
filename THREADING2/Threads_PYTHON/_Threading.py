import threading

lock = threading.Lock()

class PyThread(threading.Thread):

    def __init__(self, name):
        super(PyThread, self).__init__()
        self.name = name

    def run(self):

        for i in range(10):
            lock.acquire()
            print i, self.name,
            lock.release()

threads = []
for i in range(5):
    name = chr(i + ord('a'))
    threads.append(PyThread(name))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

for i in range(4):
     t = threading.Thread(target=worker)
     t.daemon = True  # thread dies when main thread (only non-daemon thread) exits.
     t.start()