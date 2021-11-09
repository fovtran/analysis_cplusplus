// Evgenii Rudnyi http://MatrixProgramming.com
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <time.h>

using namespace std;

extern "C" void DGESV(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);

int main(int argc,char **argv) {
  if (argc !=2)
  {
    cout << "dgesv dim" << endl;
    return 0;
  }

  clock_t t1 = clock();
  int dim=atoi(argv[1]);
  vector<double> a(dim*dim);
  vector<double> b(dim);
  vector<int> ipiv(dim);
  srand(1);              // seed the random # generator with a known value
  double maxr=(double)RAND_MAX;
  for(int r=0; r < dim; r++) {  // set a to a random matrix, i to the identity
    for(int c=0; c < dim; c++) {
      a[r + c*dim] = rand()/maxr;
    }
    b[r] = rand()/maxr;
  }
  vector<double> a1(a);
  vector<double> b1(b);
  int info;
  cout << "matrices allocated and initialised " << double(clock() - t1)/CLK_TCK << endl;
  clock_t c2 = clock();
	int one = 1;
  DGESV(&dim, &one, &*a.begin(), &dim, &*ipiv.begin(), &*b.begin(), &dim, &info);
  clock_t c3 = clock();
  cout << "dgesv is over for " << double(c3 - c2)/CLK_TCK << endl;
  cout << "info is " << info << endl;
  double eps = 0.;
  for (int i = 0; i < dim; ++i)
  {
    double sum = 0.;
    for (int j = 0; j < dim; ++j)
      sum += a1[i + j*dim]*b[j];
    eps += fabs(b1[i] - sum);
  }
  cout << "check is " << eps << endl;
  return 0;
}
