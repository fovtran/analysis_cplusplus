import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
np.set_printoptions(precision=4)

a = np.random.rand(50000).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a_dev = cl_array.to_device(queue, a)

with open("cluster.cl", 'r') as f:
    prg = cl.Program(ctx, f.read()).build()

prg.cumsum(queue, a.shape, None, a_dev.data)
print(np.cumsum(a)[:33], a_dev[:33])
