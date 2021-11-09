#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

int main()
{

  Eigen::Matrix<double, 3, 3> A;
  A << -17,  -6,  0,
       -15,   6,  14,
         9, -12,  19;

  Eigen::Matrix<double, 5, 5> B;
  B << 5, -17, -12,  16,  11,
      -4,  19,  -1,   9,  13,
       1,   3,   5,  -5,   2,
       8, -15,   5,  14, -12,
      -2,  -4,  13,  -8, -17;

  Eigen::Matrix<double, 3, 5> Q;
  Q <<   6,   5, -17,  12,   4,
       -11,  15,   8,   1,   7,
        15,  -3,   9, -19, -10;

  Eigen::RealSchur<Eigen::MatrixXd> SchurA(A);
  Eigen::MatrixXd R = SchurA.matrixT();
  Eigen::MatrixXd U = SchurA.matrixU();

  Eigen::RealSchur<Eigen::MatrixXd> SchurB(B.transpose());
  Eigen::MatrixXd S = SchurB.matrixT();
  Eigen::MatrixXd V = SchurB.matrixU();

  Eigen::MatrixXd F = (U.transpose() * Q) * V;

  Eigen::MatrixXd Y =
    Eigen::internal::matrix_function_solve_triangular_sylvester(R, S, F);

  Eigen::MatrixXd X = (U * Y) * V.transpose();

  Eigen::MatrixXd Q_calc = A * X + X * B;

  std::cout << Q_calc - Q << std::endl;
  // Should be all zeros, but instead getting:
  // 421.868  193.032 -208.273  42.7449 -3.57527
  //-1651.66 -390.314  2043.59  -1611.1 -1843.91
  //-67.4093  207.414  1168.89 -1240.54 -1650.48

  return EXIT_SUCCESS; 

}