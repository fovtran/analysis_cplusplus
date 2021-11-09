The serial code for the computation looks like

    void computeFF_cpu( float * nSamples, float * nFeatures, float ** data, float ** corr
        #pragma omp parallel for shared(corr, data)
        for( int i=0 ; i<nFeatures ; i++ )
        {
            for( int j=0 ; j<nFeatures ; j++ )
                corr[i][j] = pearsonCorr( data[i], data[j], nSamples );
        }

int main()
{
.
.
**for( int z=0 ; z<1000 ; z++ )**
computeFF_cpu( 20000, 450, data, corr );
.
.
}
This works perfectly. Now I have attempted to solve this problem with GPU. I have converted the 2D data matrix into row-major format in GPU memory and I have verified that the copy is correctly made.

The vectors are stored as a matrix of size 900000 (ie. 450*20000) in row major format. Organized as follows 
<---nSamples of f1---><---nSamples of f2 ---><---nSamples of f3--->......

My cuda code to compute cross-correlation is as follows

    // kernel for computation of ff
    __global__ void computeFFCorr(int nSamples, int nFeatures, float * dev_data, float * dev_ff)
    {
        int tid = blockIdx.x + blockIdx.y*gridDim.x;
        if( blockIdx.x == blockIdx.y )
        dev_ff[tid] = 1.0;
        else if( tid < nFeatures*nFeatures )
        dev_ff[tid] = pearsonCorrelationScore_gpu( dev_data+(blockIdx.x*nSamples), dev_data+(blockIdx.y*nSamples), nSamples );
    }

    main()
    {
    .
    .
        // Call kernel for computation of ff
**for( int z=0 ; z<1000 ; z++ )**
        computeFFCorr<<<dim3(nFeatures,nFeatures),1>>>(nSamples, nFeatures, dev_data, corr);
        //nSamples = 20000
        // nFeatures = 450
        // dev_data -> data matrix in row major form
        // corr -> result matrix also stored in row major
    .
    .
    }