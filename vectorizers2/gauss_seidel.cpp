e if(k==2)
        {
            ygs=(3.0-2.0*y-2.0*x)/2.0;
            return ygs;
        }
    else if(k==3)
        {
            zgs=-2.0*y-x;
            return zgs;
        }
}
int main() 
{
    FILE *fp;
    fp=fopen("Jacobi-Gauss-Siedel.2.txt", "w");
    double x=0.0, y=0.0, z=0.0, xj, yj, zj, xgs, ygs, zgs;
    int i;
    for(i=0; i<3; i++)
        {
            xj=jacobi(1,x,y,z);
            yj=jacobi(2,x,y,z);
            zj=jacobi(3,x,y,z);
            x=xj, y=yj, z=zj;
            fprintf(fp, "xj= %lf   \tyj=%lf   \tzj=%lf\n", x, y, z);
        }
    for(i=0; i<3; i++)
        {
            xgs=jacobi(1,x,y,z);
            ygs=jacobi(2,xgs,y,z);
            zgs=jacobi(3,xgs,ygs,z);
            fprintf(fp, "xgs= %lf   \tyj=%lf   \tzj=%lf\n", xgs, ygs, zgs);
        }
    fclose(fp);
    return 0;
}