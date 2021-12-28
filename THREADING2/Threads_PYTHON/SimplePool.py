from multiprocessing.dummy import Pool as ThreadPool 

pool = ThreadPool(4) 
results = pool.map(my_function, my_array)

# Which is the multithreaded version of:
results = []
for item in my_array:
    results.append(my_function(item))