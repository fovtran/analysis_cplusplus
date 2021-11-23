__kernel void cumsum(__global float *a)
{
    int gid = get_global_id(0);
    int n = get_global_size(0);

    for (int i = 1; i < n; i <<= 1)
        if (gid & i)
            a[gid] += a[(gid & -i) - 1];
}
