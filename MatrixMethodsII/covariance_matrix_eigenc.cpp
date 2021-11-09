// covariance matrix Eigen
#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "common.hpp"

int test_calcCovarMatrix()
{
	// reference: https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
					{ -3.6f, 9.2f, 0.5f, 7.2f },
					{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);

	fprintf(stderr, "source matrix:\n");
	std::cout << m << std::endl;

	fprintf(stdout, "\nEigen implement:\n");
	const int nsamples = rows;
	float scale = 1. / (nsamples /*- 1*/);

	Eigen::MatrixXf mean = m.colwise().mean();
	std::cout << "print mean: " << std::endl << mean << std::endl;

	Eigen::MatrixXf tmp(rows, cols);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			tmp(y, x) = m(y, x) - mean(0, x);
		}
	}
	//std::cout << "tmp: " << std::endl << tmp << std::endl;

	Eigen::MatrixXf covar = (tmp.adjoint() * tmp) /*/ float(nsamples - 1)*/;
	std::cout << "print covariance matrix: " << std::endl << covar << std::endl;

	return 0;
}
