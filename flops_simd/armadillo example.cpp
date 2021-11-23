#include <armadillo>
#include <iostream>

using arma::cx_mat;
using arma::sp_cx_mat;

int main()
{
    using namespace std::complex_literals;
    cx_mat A = {{3, 0, 5.0 + 2i},
                {0, 4, 0},
                {5.0 - 2i, 0, 13}};
    sp_cx_mat B(A);

    std::cout << "Dense non-symmetric:\n"
              << eig_gen(A)
              << "\nDense symmetric:\n"
              << eig_sym(A)
              << "\nSparse non-symmetric:\n"
              << eigs_gen(B, 1, "lm")
              << eigs_gen(B, 1, "sm") // changing the 1 to a 2 results in an error here
              << std::endl;

    return 0;
}