// SVD eigen
#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "common.hpp"

int test_SVD()
{
	//std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
	//				{ -3.6f, 9.2f, 0.5f, 7.2f },
	//				{ 4.3f, 1.3f, 9.4f, -3.4f },
	//				{ 6.4f, 0.1f, -3.7f, 0.9f } };
	//const int rows{ 4 }, cols{ 4 };

	//std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
	//				{ -3.6f, 9.2f, 0.5f, 7.2f },
	//				{ 4.3f, 1.3f, 9.4f, -3.4f } };
	//const int rows{ 3 }, cols{ 4 };

	std::vector<std::vector<float>> vec{ { 0.68f, 0.597f },
					{ -0.211f, 0.823f },
					{ 0.566f, -0.605f } };
	const int rows{ 3 }, cols{ 2 };

	std::vector<float> vec_;
	for (int i = 0; i < rows; ++i) {
		vec_.insert(vec_.begin() + i * cols, vec[i].begin(), vec[i].end());
	}
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);

	fprintf(stderr, "source matrix:\n");
	std::cout << m << std::endl;

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
	Eigen::MatrixXf singular_values = svd.singularValues();
	Eigen::MatrixXf left_singular_vectors = svd.matrixU();
	Eigen::MatrixXf right_singular_vectors = svd.matrixV();

	fprintf(stderr, "singular values:\n");
	print_matrix(singular_values.data(), singular_values.rows(), singular_values.cols());
	fprintf(stderr, "left singular vectors:\n");
	print_matrix(left_singular_vectors.data(), left_singular_vectors.rows(), left_singular_vectors.cols());
	fprintf(stderr, "right singular vecotrs:\n");
	print_matrix(right_singular_vectors.data(), right_singular_vectors.rows(), right_singular_vectors.cols());

	return 0;
}
