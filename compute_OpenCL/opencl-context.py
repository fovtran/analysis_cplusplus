import pyopencl as cl
plat = cl.get_platforms()
devices = plat[0].get_devices()
ctx = cl.Context([devices[1]])
ctx.get_info(cl.context_info.DEVICES)