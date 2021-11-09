// inverse matrix Eigen
#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "common.hpp"
 
template<typename _Tp>
void print_matrix(const _Tp* data, const int rows, const int cols)
{
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			fprintf(stderr, "  %f  ", static_cast<float>(data[y * cols + x]));
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

int test_inverse_matrix()
{
	std::vector<float> vec{ 5, -2, 2, 7, 1, 0, 0, 3, -3, 1, 5, 0, 3, -1, -9, 4 };
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}

	Eigen::Map<Eigen::MatrixXf> map(vec.data(), 4, 4);
	Eigen::MatrixXf inv = map.inverse();

	fprintf(stderr, "source matrix:\n");
	print_matrix<float>(vec.data(), N, N);
	fprintf(stderr, "eigen inverse matrix:\n");
	print_matrix<float>(inv.data(), N, N);

	return 0;
}
