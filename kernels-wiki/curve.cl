#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__private int k = 2; // I need k to be an int, because I want to use as a counter
__private double s = 18;
__private double a = 1;

a = a/(double)k; // just to show, that I make in-place typecasting of k
a = k+1;
k = (int)a; //to show that I store k in a double buffer in an intermediate-step
if ((k-1)==2)
{
//    k = 3;
    s = pow(s/(double)(k-1),0.5);
	s = sqrt(s/(double)(k-1))
}