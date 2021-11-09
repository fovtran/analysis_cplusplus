const sampler_t mSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP|
                           CLK_FILTER_NEAREST;


__kernel void computeHistogram(read_only image2d_t input, __global int* rOutput,
                               __global int* gOutput, __global int* bOutput)
{

    int2 coords = {get_global_id(0), get_global_id(1)};

    float4 sample = read_imagef(input, mSampler, coords);

    uchar rbin = floor(sample.x * 255.0f);
    uchar gbin = floor(sample.y * 255.0f);
    uchar bbin = floor(sample.z * 255.0f);

    rOutput[rbin]++;
    gOutput[gbin]++;
    bOutput[bbin]++;
}