from multiprocessing import freeze_support, Queue, Process, Pool, cpu_count
import string
import time
from random import seed, choice
import asyncio
import concurrent.futures

seed(123)
numcores = cpu_count()
output = Queue()

def rand_string2(length):
	rand_str = ''.join(choice( string.ascii_lowercase + string.ascii_uppercase + string.digits)	for i in range(length))
	return rand_str

def rand_string(length, output):
	rand_str = ''.join(choice( string.ascii_lowercase + string.ascii_uppercase + string.digits)	for i in range(length))
	output.put(rand_str)
	return True

def funct2(ab):
	begin_time = time.time()
	time.sleep(0.1)

arg = (4,output)

def uno():
	processes = [Process(target=rand_string, args=arg) for x in range(numcores)]
	for p in processes:
		p.start()

	for p in processes:
		p.join()
		p.terminate()
	results = [output.get() for p in processes]
	print (results)
	print (output.qsize())

def dos():
	pool = Pool(processes = numcores)
	it = pool.imap(rand_string2, [4,4,4,4])
	results = [x for x in it]
	pool.close()
	pool.join()
	print(results)

if __name__ == '__main__':
	freeze_support()
	uno()
	dos()