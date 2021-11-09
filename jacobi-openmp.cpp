void ROTATE(MatrixXd &a, int i, int j, int k, int l, double s, double tau) {
	double g,h;
	g=a(i,j);
	h=a(k,l);
	a(i,j)=g-s*(h+g*tau);
	a(k,l)=h+s*(g-h*tau);
}

void jacobi(int n, MatrixXd &a, MatrixXd &v, VectorXd &d ){
	int j,iq,ip,i;
	double tresh,theta,tau,t,sm,s,h,g,c;

	VectorXd b(n);
	VectorXd z(n);

	v.setIdentity();
	z.setZero();

	#pragma omp parallel for
	for (ip=0;ip<n;ip++)
	{
	    d(ip)=a(ip,ip);
	    b(ip)=d(ip);
	}

	for (i=0;i<50;i++)
	{
	    sm=0.0;
	    for (ip=0;ip<n-1;ip++)
	    {
	        #pragma omp parallel for reduction (+:sm)
	        for (iq=ip+1;iq<n;iq++)
	            sm += fabs(a(ip,iq));
	    }
	    if (sm == 0.0) {
	        break;
	    }

	    if (i < 3)
	    tresh=0.2*sm/(n*n);
	    else
	    tresh=0.0;

	    #pragma omp parallel for private (ip,g,h,t,theta,c,s,tau)
	    for (ip=0;ip<n-1;ip++)
	    {
	    //#pragma omp parallel for private (g,h,t,theta,c,s,tau)
	        for (iq=ip+1;iq<n;iq++)
	        {
	            g=100.0*fabs(a(ip,iq));
	            if (i > 3 && (fabs(d(ip))+g) == fabs(d[ip]) && (fabs(d[iq])+g) == fabs(d[iq]))
	            a(ip,iq)=0.0;
	            else if (fabs(a(ip,iq)) > tresh)
	            {
	                h=d(iq)-d(ip);
	                if ((fabs(h)+g) == fabs(h))
	                {
	                    t=(a(ip,iq))/h;
	                }
	                else
	                {
	                    theta=0.5*h/(a(ip,iq));
	                    t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
	                    if (theta < 0.0)
	                    {
	                        t = -t;
	                    }
	                    c=1.0/sqrt(1+t*t);
	                    s=t*c;
	                    tau=s/(1.0+c);
	                    h=t*a(ip,iq);

	                   #pragma omp critical
	                    {
	                    z(ip)=z(ip)-h;
	                    z(iq)=z(iq)+h;
	                    d(ip)=d(ip)-h;
	                    d(iq)=d(iq)+h;
	                    a(ip,iq)=0.0;

	                    for (j=0;j<ip;j++)
	                        ROTATE(a,j,ip,j,iq,s,tau);
	                    for (j=ip+1;j<iq;j++)
	                        ROTATE(a,ip,j,j,iq,s,tau);
	                    for (j=iq+1;j<n;j++)
	                        ROTATE(a,ip,j,iq,j,s,tau);
	                    for (j=0;j<n;j++)
	                        ROTATE(v,j,ip,j,iq,s,tau);
	                    }

	                }
	            }
	        }
	    }
}
