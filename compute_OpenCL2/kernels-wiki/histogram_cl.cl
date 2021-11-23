#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#pragma OPENCL EXTENSION cl_khr_fp16 : enable
//#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
//#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode : disable
#pragma OPENCL SELECT_ROUNDING_MODE rte

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|CLK_FILTER_NEAREST|CLK_ADDRESS_CLAMP_TO_EDGE;

void histogram(read_only image2d_t input, write_only image2d_t outputImage){
	int2 dimensions = get_image_dim(input);
	int width = dimensions.x, height = dimensions.y;	
	int x = get_global_id(0), y = get_global_id(1);
	float4 pixel = read_imagef(input, sampler, (int2)(x, y));
	float4 transformedPixel;
	float luminance = dot((float4)(1/3.f, 1/3.f, 1/3.f, 0), pixel);

	float factor = 1.0f;
	uchar rbin = pixel.x * factor;
	uchar gbin = pixel.y * factor;
	uchar bbin = pixel.z * factor;
	uchar abin = pixel.w * factor;

	//transformedPixel = pixel+((float4)(rbin, gbin, bbin, abin));
	transformedPixel = pixel+luminance * ((float4)(1.0f,0.4f,1.0f,0.0f)); 
	write_imagef(outputImage, (int2)(x, y), transformedPixel);
}

__kernel void pass(read_only image2d_t input, write_only image2d_t outputImage)
{
	histogram(input, outputImage);
}