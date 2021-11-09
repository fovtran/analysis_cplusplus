// norm positive infinity and norm L implemented in eigen
#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "common.hpp"

int test_norm()
{
	fprintf(stderr, "test norm with eigen:\n");
	// 1. vector
	std::vector<float> vec1{ -2, 3, 1 };
	Eigen::VectorXf v(vec1.size());
	for (int i = 0; i < vec1.size(); ++i) {
		v[i] = vec1[i];
	}

	double value = v.lpNorm<Eigen::Infinity>();
	fprintf(stderr, "vector: Inf: %f\n", value);
	value = v.lpNorm<1>();
	fprintf(stderr, "vector: L1: %f\n", value);
	value = v.norm(); // <==> sqrt(v.squaredNorm()) <==> v.lpNorm<2>()
	fprintf(stderr, "vector: L2: %f\n", value);

	// 2. matrix
	std::vector<float> vec2{ -3, 2, 0, 5, 6, 2, 7, 4, 8 };
	const int row_col{ 3 };
	Eigen::Map<Eigen::MatrixXf> m(vec2.data(), row_col, row_col);

	value = m.lpNorm<Eigen::Infinity>();
	fprintf(stderr, "matrix: Inf: %f\n", value);
	value = m.lpNorm<1>();
	fprintf(stderr, "matrix: L1: %f\n", value);
	value = m.norm();
	fprintf(stderr, "matrix: L2: %f\n", value);

	return 0;
}
