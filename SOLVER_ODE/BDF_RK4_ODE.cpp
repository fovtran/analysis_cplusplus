BDF and RK4 methods 
#include<stdio.h>
#include<math.h>
#include<iostream>

#define MAX_N 10000
using namespace std;
static   double F(double, double);
static   double S(double, double);
static   double Fprime(double, double);
static   double Sprime(double, double);

int main()
{
    double K1_1,K2_1,K3_1,K4_1,K1_2,K2_2,K3_2,K4_2,W[MAX_N],V[MAX_N],H,T;
    double A = 0.0;
    double B = 5.0;
    int N = 10000;
    int I;

    cout.setf(ios::fixed,ios::floatfield);
    cout.precision(9);

    H = (B - A) / N;
    T = A;
    W[0] = 0.5;
    V[0] = 0.0;

    for (I=1; I<=3; I++)
    {
         K1_1 = H*F(W[I-1], V[I-1]);
         K1_2 = H*S(W[I-1], V[I-1]);

         K2_1 = H*F(W[I-1] + K1_1/2.0, V[I-1] + K1_2/2.0);
         K2_2 = H*S(W[I-1] + K1_1/2.0, V[I-1] + K1_2/2.0);

         K3_1 = H*F(W[I-1] + K2_1/2.0, V[I-1] + K2_2/2.0);
         K3_2 = H*S(W[I-1] + K2_1/2.0, V[I-1] + K2_2/2.0);

         K4_1 = H*F(W[I-1] + K3_1, V[I-1] + K3_2);
         K4_2 = H*S(W[I-1] + K3_1, V[I-1] + K3_2);

         W[I] = W[I-1] + 1/6.0*(K1_1 + 2.0*K2_1 + 2.0*K3_1 + K4_1);
         V[I] = V[I-1] + 1/6.0*(K1_2 + 2.0*K2_2 + 2.0*K3_2 + K4_2);

         T = A + I * H;

         cout <<"At time "<< T <<" the solution = "<< W[I] << endl;
    }
    //BDF order 4 to get the rest of the points
      for(I = 4; I <= N; I++)
      {
          //Newton Raphson method to get the values of W[I],V[I] for the implicit BDF
          double W_temp = W[I-1];
          double V_temp = V[I-1];
          double tol = 1e-14;
          double error = tol + 1;
          int iteration = 0;

          //Checking tolerance, the denominator not being too small, and a reasonable number of iterations
          while (error > tol && fabs(Fprime(W_temp, V[I-1]))>1e-14 && iteration < 1000)
          {
            W[I] = W_temp - F(W_temp, V[I-1])/Fprime(W_temp, V[I-1]);

            error = fabs(W[I] - W_temp);
            W_temp = W[I];
            iteration++;

          }
          iteration = 0;

          while (error > tol && Sprime(W[I-1], V_temp)>1e-14 && iteration < 1000)
          {
            V[I] = V_temp - S(W[I-1], V_temp)/Sprime(W[I-1], V_temp);

            error = fabs(V[I] - V_temp);
            V_temp = V[I];
            iteration++;

          }

          //BDF order 4
          W[I] = (48.0*W[I-1] - 36.0*W[I-2] + 16.0*W[I-3] - 3.0*W[I-4] + 12.0*H*F(W[I],V[I]))/25.0;
          V[I] = (48.0*V[I-1] - 36.0*V[I-2] + 16.0*V[I-3] - 3.0*V[I-4] + 12.0*H*S(W[I],V[I]))/25.0;

          T = A + I * H;

          cout <<"At time "<< T <<" the solution = "<< W[I] << endl;
      }

    return 0;
}

/*  First incremental function  */
double F(double y1, double y2)
{
   double f; 

   f = 40.0*y1 - 2.0*y2 + 40.0*pow(y2,2) - 100.0*y1*pow(y2,2) + 160.0*pow(y1,2)*pow(y2,4) + 100.0*pow(y1,2)*pow(y2,2) - 180.0*y1*pow(y2,6) + 180.0*y1*pow(y2,4) -240.0*pow(y1,4)*pow(y2,2) +100.0*pow(y1,3)*pow(y2,4) + 220.0*pow(y1,2)*pow(y2,6) - 180.0*y1*pow(y2,8) + 4.0*y1*y2 - 60*pow(y2,12) - 20.0*pow(y1,7) - 20.0*pow(y2,14) - 60.0*pow(y1,5)*pow(y2,2) + 180.0*pow(y1,4)*pow(y2,4) - 120.0*pow(y1,3)*pow(y2,6) - 120.0*pow(y1,2)*pow(y2,8) + 180.0*y1*pow(y2,10) + 100.0*pow(y1,6)*pow(y2,2) - 180.0*pow(y1,5)*pow(y2,4) + 100.0*pow(y1,4)*pow(y2,6) + 100.0*pow(y1,3)*pow(y2,8) - 180.0*pow(y1,2)*pow(y2,10) + 100.0*y1*pow(y2,12) + 140.0*pow(y2,8) - 20.0*pow(y2,6) - 100.0*pow(y1,3) + 80.0*pow(y1,5) + 20.0*pow(y2,10) - 4.0*pow(y2,3) - 100.0*pow(y2,4);
   return f;
}
double Fprime(double y1, double y2)
{
  double fprime;

  fprime = 40.0 - 100.0*pow(y2,2) + 480.0*pow(y1,2)*pow(y2,2) - 320.0*y1*pow(y2,4) + 200.0*y1*pow(y2,2) - 180.0*pow(y2,6) + 180.0*pow(y2,4) - 960.0*pow(y1,3)*pow(y2,2) + 300.0*pow(y1,2)*pow(y2,4) + 440.0*y1*pow(y2,6) - 180.0*pow(y2,8) + 4.0*y2 - 140.0*pow(y1,6) - 300.0*pow(y1,4)*pow(y2,2) + 720.0*pow(y1,3)*pow(y2,4) - 360.0*pow(y1,2)*pow(y2,6) - 240.0*y1*pow(y2,8) + 180.0*pow(y2,10) + 600.0*pow(y1,5)*pow(y2,2) - 900.0*pow(y1,4)*pow(y2,4) + 400.0*pow(y1,3)*pow(y2,6) + 300.0*pow(y1,2)*pow(y2,8) - 360.0*y1*pow(y2,10) + 100.0*pow(y2,12) - 300.0*pow(y1,2) + 400.0*pow(y1,4);
  return fprime;
}



/*  Second incremental function  */
double S(double y1, double y2)
{
   double s; 

   s = 2.0*y1 + 40.0*y2 - 2.0*pow(y2,2) - 100.0*pow(y1,2)*y2 + 200.0*y1*pow(y2,3) - 320.0*pow(y1,3)*pow(y2,3) + 420.0*pow(y1,2)*pow(y2,5) + 160.0*pow(y1,2)*pow(y2,3) - 200.0*y1*pow(y2,7) - 320.0*y1*pow(y2,5) - 60.0*pow(y1,4)*pow(y2,3) + 240.0*pow(y1,3)*pow(y2,5) - 360.0*pow(y1,2)*pow(y2,7) + 240.0*y1*pow(y2,9) + 120.0*pow(y1,5)*pow(y2,3) - 300.0*pow(y1,4)*pow(y2,5) + 400.0*pow(y1,3)*pow(y2,7) - 300.0*pow(y1,2)*pow(y2,9) + 120.0*y1*pow(y2,11) - 20.0*pow(y1,6)*y2 + 80.0*pow(y1,4)*y2 - 100.0*pow(y2,3) - 20.0*pow(y2,13) + 20.0*pow(y2,9) + 140.0*pow(y2,7) - 60.0*pow(y2,11) - 20.0*pow(y2,5);
   return s;
}
double Sprime(double y1, double y2)
{
  double fprime;

  fprime = 40.0 - 4.0*y2 - 100.0*pow(y1,2) + 600.0*y1*pow(y2,2) - 960.0*pow(y1,3)*pow(y2,2) + 2100.0*pow(y1,2)*pow(y2,4) + 480.0*pow(y1,2)*pow(y2,2) - 1400.0*y1*pow(y2,6) - 1600.0*y1*pow(y2,4) - 180*pow(y1,4)*pow(y2,2) + 1200.0*pow(y1,3)*pow(y2,4) - 2520.0*pow(y1,2)*pow(y2,6) + 2160.0*y1*pow(y2,8) + 360*pow(y1,5)*pow(y2,2) - 1500.0*pow(y1,4)*pow(y2,4) + 2800.0*pow(y1,3)*pow(y2,6) - 2700.0*pow(y1,2)*pow(y2,8) + 1320.0*y1*pow(y2,11) - 20.0*pow(y1,6) + 80.0*pow(y1,4) - 300.0*pow(y2,2) - 260.0*pow(y2,12) + 180.0*pow(y2,8) + 980.0*pow(y2,6) - 660.0*pow(y2,10) - 100.0*pow(y2,4);
  return fprime;
}