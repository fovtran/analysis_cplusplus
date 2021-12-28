import Queue
import threading
import urllib2

worker_data = ['http://google.com', 'http://yahoo.com', 'http://bing.com']

#load up a queue with your data, this will handle locking
q = Queue.Queue()
for url in worker_data:
    q.put(url)

#define a worker function
def worker(queue):
    queue_full = True
    while queue_full:
        try:
            #get your data off the queue, and do some work
            url= queue.get(False)
            data = urllib2.urlopen(url).read()
            print len(data)

        except Queue.Empty:
            queue_full = False

#create as many threads as you want
thread_count = 5
for i in range(thread_count):
    t = threading.Thread(target=worker, args = (q,))
    t.start()