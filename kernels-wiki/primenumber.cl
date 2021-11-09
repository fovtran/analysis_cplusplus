__kernel void dataParallel(__global int* A)
{
    A[0]=2;
    A[1]=3;
    A[2]=5;
    int pnp;//pnp=probable next prime
    int pprime;//previous prime
    int i,j;
    for(i=3;i<5000;i++)
    {
        j=0;
        pprime=A[i-1];
        pnp=pprime+2;
        while((j<i) && A[j]<=sqrt((float)pnp))
        {
            if(pnp%A[j]==0)
                {
                    pnp+=2;
                    j=0;
                }
            j++;

    }
    A[i]=pnp;

    }
}
