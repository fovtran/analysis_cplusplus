/*
FFT convolution of real signals is very easy. Multiplication in the frequency domain is equivalent to convolution in the time domain. The important thing to remember however, is that you are multiplying complex numbers, and therefore, you must to a "complex multiplication".

So the steps are:
Do an FFT of your filter kernel,
Do an FFT of your "dry" signal.
do a complex multiply of the two spectra
Perform the inverse FFT of this new spectrum.
Of course if you want to do continuous processing of lenghty signals, then you will need to use the overlap-add or overlap-save method.
If you are using real signals only, on an Intel format (little endian) machine, you can use Surreall FFT plus the multiply function I give below that is in the correct format for the data order.
*/

#define F_TYPE float;

void CompMulR(F_TYPE a[],F_TYPE b[],F_TYPE res[],long n)
  {
  long i;
  for(i=0;i<n;i++)
    {
    if(i<2)
      res[i]=a[i]*b[i]; // DC & NQ are not complex
    else
      {
      if((i&1)==0)
        {//real
        res[i]= a[i]*b[i] - a[i+1]*b[i+1];
        }
      else
        {//img.
        res[i]= a[i]*b[i-1] + a[i-1]*b[i];
        }
      }
    }
  }

#define N 1024
F_TYPE TimeDom [N];
F_TYPE FrqDom [N];
F_TYPE FrqDom2 [N];
F_TYPE Twidds[N/4];

InitFFT(N, Twidds); //Initialse the twiddle factors
FFT(TimeDom,FrqDom,N,Twidds); // do a forward transform
CompMulR(TimeDom,FrqDom,FrqDom2[],N);//do a complex multiply 
IFFT(TimeDom,FrqDom2,N,Twidds); // do an inverse transform 
//TimeDom now holds the convolution of the 2 signals