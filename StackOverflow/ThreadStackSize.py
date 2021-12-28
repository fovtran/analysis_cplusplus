import sys
import threading
import pdb; pdb.set_trace()

class SomeCallable:
    def __call__(self):
        try:
            self.recurse(99900)
        except RecursionError:
            print("Booh!")
        else:
            print("Hurray!")
    def recurse(self, n):
        if n > 0:
            self.recurse(n-1)

SomeCallable()() # recurse in current thread

# recurse in greedy thread
sys.setrecursionlimit(100000)
threading.stack_size(0x2000000)
t = threading.Thread(target=SomeCallable())
t.start()
t.join()
