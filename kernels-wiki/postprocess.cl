__kernel void postProcess(__global uchar * input, __global uchar * output)
{
    int resultImgSize=1024;
    int pixelX=get_global_id(0)%resultImgSize; // 1-D id list to 2D workitems(each process a single pixel)
    int pixelY=get_global_id(0)/resultImgSize;
    int imgW=resultImgSize;
    int imgH=resultImgSize;


    float kernelx[3][3] = {{-1, 0, 1}, 
                           {-2, 0, 2}, 
                           {-1, 0, 1}};
    float kernely[3][3] = {{-1, -2, -1}, 
                           {0,  0,  0}, 
                           {1,  2,  1}};

    // also colors are separable
    int magXr=0,magYr=0; // red
    int magXg=0,magYg=0;
    int magXb=0,magYb=0;

    // Sobel filter
    // this conditional leaves 10-pixel-wide edges out of processing
    if( (pixelX<imgW-10) && (pixelY<imgH-10) && (pixelX>10) && (pixelY>10) )
    { 
        for(int a = 0; a < 3; a++)
        {
            for(int b = 0; b < 3; b++)
            {            
                int xn = pixelX + a - 1;
                int yn = pixelY + b - 1;

                int index = xn + yn * resultImgSize;
                magXr += input[index*4] * kernelx[a][b];
                magXg += input[index*4+1] * kernelx[a][b];
                magXb += input[index*4+2] * kernelx[a][b];
                magYr += input[index*4] * kernely[a][b];
                magYg += input[index*4+1] * kernely[a][b];
                magYb += input[index*4+2] * kernely[a][b];
            }
         }
    }

    // magnitude of x+y vector
    output[(pixelX+pixelY*resultImgSize)*4]  =sqrt((float)(magXr*magXr + magYr*magYr)) ;
    output[(pixelX+pixelY*resultImgSize)*4+1]=sqrt((float)(magXg*magXg + magYg*magYg)) ;
    output[(pixelX+pixelY*resultImgSize)*4+2]=sqrt((float)(magXb*magXb + magYb*magYb)) ;
    output[(pixelX+pixelY*resultImgSize)*4+3]=255;

}