__global__ void PostprocessKernel( uchar4* dst, unsigned int imgWidth, unsigned int imgHeight )
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bw = blockDim.x;
    unsigned int bh = blockDim.y;
 
    // Non-normalized U, V coordinates of input texture for current thread.
    unsigned int u = ( bw * blockIdx.x ) + tx;
    unsigned int v = ( bh * blockIdx.y ) + ty;
 
    // Early-out if we are beyond the texture coordinates for our texture.
    if ( u > imgWidth || v > imgHeight ) return;
 
    // The 1D index in the destination buffer.
    unsigned int index = ( v * imgWidth ) + u;
     
    float4 tempColor = make_float4(0, 0, 0, 1);
    for ( int i = 0; i < FILTER_SIZE; ++i )
    {
        // Fetch a texture element from the source texture.
        uchar4 color = tex2D( texRef, u + indexOffsetsU[i], v + indexOffsetsV[i] );
 
        tempColor.x += color.x * kernelFilter[i];
        tempColor.y += color.y * kernelFilter[i];
        tempColor.z += color.z * kernelFilter[i];
    }
 
    // Store the processed color in the destination buffer.
    dst[index] = make_uchar4( 
        Clamp<unsigned char>(tempColor.x * invScale + offset, 0.0f, 255.0f), 
        Clamp<unsigned char>(tempColor.y * invScale + offset, 0.0f, 255.0f), 
        Clamp<unsigned char>(tempColor.z * invScale + offset, 0.0f, 255.0f), 
        1
    );
}