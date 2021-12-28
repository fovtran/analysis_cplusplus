from threading import Thread
from multiprocessing import Process, Lock, Pool, freeze_support, Queue, cpu_count, TimeoutError
from math import pow, sqrt
import time
from PyQt5 import QtCore

#from numpy import genfromtxt
#my_data = genfromtxt('my_file.csv', delimiter=',')
#import pandas as pd
#df=pd.read_csv('myfile.csv', sep=',',header=None)

class Foo1():
	results = []

class Foo2(QtCore.QObject):
	results = []

results = Foo1().results
results = Foo2().results
results = []

M = lambda n: [pow(y,3) for y in range(n)]
def ta(q): [results.append(sqrt(x)) for x in q]
def tb(q): return [sqrt(x) for x in q]

if __name__ == "__main__":
	freeze_support()

	time.sleep(1) # Settle time 
	a = time.time()
	_thread = Thread( target=ta, args=(M(1000000),) )
	_thread.start()
	_thread.join()
	b = time.time()
	print(results[1:10])

	time.sleep(1) # Settle time
	results = []
	c = time.time()
	pool = Pool(processes=4)
	#async_result = pool.apply_async(ta, (M(1000001),))
	async_result = pool.map_async(tb, (M(1000000),), chunksize=250000)
	#return_val = async_result.get()
	d = time.time()
	#print(return_val[1:10])
	print(results[1:10])

	print("Result A: ", b-a)
	print("Result B: ", d-c)
	#print(results==return_val)

