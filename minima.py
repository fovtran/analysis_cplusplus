import pyopencl as cl
import numpy as np

platform = cl.get_platforms()[0]
devs = platform.get_devices()
device1 = devs[0]
h_buffer = np.empty((10,10)).astype(np.float64)
mf = cl.mem_flags
ctx = cl.Context([device1])
Queue1 = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
Queue2 = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags
m_dic = {0:mf.READ_ONLY,1:mf.WRITE_ONLY,2:mf.READ_WRITE}

fi = open('Minima.cl', 'r')
fstr = "".join(fi.readlines())
prg = cl.Program(ctx, fstr).build()
knl = prg.init_z
knl.set_scalar_arg_dtypes([None,])
knl_root = prg.root
knl_root.set_scalar_arg_dtypes([None,])

def f():
    d_buffer =  cl.Buffer(ctx,m_dic[2], int(10 * 10  * 8))
    knl.set_args(d_buffer)
    knl_root.set_args(d_buffer)
    a = cl.enqueue_nd_range_kernel(Queue2,knl,(10,10),None)
    b = cl.enqueue_nd_range_kernel(Queue2,knl_root,(10,10),None, wait_for = [a,])
    cl.enqueue_copy(Queue1,h_buffer,d_buffer,wait_for=[b,])
    return h_buffer
a = f()
print( a[0,0] ) # Getting the result on the host.
