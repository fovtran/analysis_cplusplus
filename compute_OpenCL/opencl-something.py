import pyopencl as cl
plat = cl.get_platforms()
plat[0].get_devices()