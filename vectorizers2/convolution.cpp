void convolution_example()
{
	for (size_t k = 0; k < nkernels; ++k) {
	    for (size_t y = 0; y < out_height; ++y) {
	        for (size_t x = 0; x < out_width; ++x) {
	            for (size_t c = 0; c < depth; ++c) {
	                for (size_t ky = 0; ky < kernel_height; ++ky) {
	                    for (size_t kx = 0; kx < kernel_width; ++kx) {

	                        out[y][x][k] += weights[ky][kx][c][k] * in[y + ky][x + kx][c];

	                    }
	                }
	            }

	            out[y][x][k] += biases[k];
				
	        }
	    }
	}
}
