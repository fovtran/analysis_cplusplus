// g++ -I./<path-to-eigen-dir> -std=c++11 -Wall -O3 -fPIC  -c example.cpp -o example

#include <Eigen/Dense>
#include <iostream>

// Declare and initialize matrices
Eigen::MatrixXi A(2, 3);
A <<  1, 1, 1,
      0, 1, 0;

Eigen::MatrixXi B(3, 2);
B << 2, 0,
     3, 0,
     1, 2;

Eigen::MatrixXi C = A + B.transpose();

std::cout<<C<<std::endl;
