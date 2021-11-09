// eigenvals and eigenvect eigen
#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "common.hpp"

int test_eigenvalues_eigenvectors()
{
    std::vector<float> vec{ 1.23f, 2.12f, -4.2f,
        2.12f, -5.6f, 8.79f,
        -4.2f, 8.79f, 7.3f };
    const int N{ 3 };

    fprintf(stderr, "source matrix:\n");
    print_matrix(vec.data(), N, N);
    fprintf(stderr, "\n");

    Eigen::Map<Eigen::MatrixXf> m(vec.data(), N, N);

    Eigen::EigenSolver<Eigen::MatrixXf> es(m);
    Eigen::VectorXf eigen_values = es.eigenvalues().real();
    fprintf(stderr, "eigen values:\n");
    print_matrix(eigen_values.data(), N, 1);
    Eigen::MatrixXf eigen_vectors = es.eigenvectors().real();
    fprintf(stderr, "eigen vectors:\n");
    print_matrix(eigen_vectors.data(), N, N);

    return 0;
}  
