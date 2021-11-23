#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void init_z(__global double * buffer)
{
    __private int x = get_global_id(0);
    __private int y = get_global_id(1);
    //w,h
    __private int w_y = get_global_size(1);
    __private int address = x*w_y+y;
    //h,w
    __private double init = 3.0;
    buffer[address]=init;
}

__kernel void root(__global double * buffer)
{
    __private int x = get_global_id(0);
    __private int y = get_global_id(1);
    //w,h
    __private int w_y = get_global_size(1);
    __private int address = x*w_y+y;
    //h,w
    __private double value = 18;
    __private int k;
    __private double out;
    k = (int) buffer[address];
  //k = 3;  If this line is uncommented, the result will be exact.
    out = pow(value/(double)(k-1), 0.5);
    buffer[address] = out;
}