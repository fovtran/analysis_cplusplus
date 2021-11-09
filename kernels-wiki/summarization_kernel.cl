/*
 *  Copyright (c) 2012 Matthew Arsenault
 *  Copyright (c) 2012 Rensselaer Polytechnic Institute
 *  This file is part of Milkway@Home.
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DOUBLEPREC
  #error DOUBLEPREC not defined
#endif

#if DOUBLEPREC
  /* double precision is optional core feature in 1.2, not an extension */
  #if __OPENCL_VERSION__ < 120
    #if cl_khr_fp64
      #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif cl_amd_fp64
      #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #else
      #error Missing double precision extension
    #endif
  #endif
#endif /* DOUBLEPREC */

#if DOUBLEPREC
typedef double real;
typedef double2 real2;
typedef double4 real4;
#else
typedef float real;
typedef float2 real2;
typedef float4 real4;
#endif /* DOUBLEPREC */


inline real2 kahanReduction(real2 inOut, real2 in)
{
    real correctedNextTerm, newSum;

    correctedNextTerm = in.x + (in.y + inOut.y);
    newSum = inOut.x + correctedNextTerm;
    inOut.y = correctedNextTerm - (newSum - inOut.x);
    inOut.x = newSum;

    return inOut;
}

/* It's not really important if this kernel is fast */
__kernel void summarization(__global real2* restrict results,
                            __global const real2* restrict buffer,

                            const uint nElements,
                            const uint bufferOffset)
{
    __local real2 sdata[128];
    uint gid = get_global_id(0);
    int lid = get_local_id(0);

    sdata[lid] = (gid < nElements) ? buffer[bufferOffset + gid] : 0.0;
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    for (int offset = get_local_size(0) / 2; offset > 0; offset >>= 1)
    {
        if (lid < offset)
        {
            sdata[lid] = kahanReduction(sdata[lid], sdata[lid + offset]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        results[get_group_id(0)] = sdata[0];
    }
}
