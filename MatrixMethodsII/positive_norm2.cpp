// norm positive infinity and norm L implemented in C++
#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "common.hpp"

// Find the norm
typedef enum Norm_Types_ {
	 Norm_INT = 0, // infinity
	Norm_L1, // L1
	Norm_L2 // L2
} Norm_Types;

template<typename _Tp>
int norm(const std::vector<std::vector<_Tp>>& mat, int type, double* value)
{
	*value = 0.f;

	switch (type) {
		case Norm_INT: {
			for (int i = 0; i < mat.size(); ++i) {
				for (const auto& t : mat[i]) {
					*value = std::max(*value, (double)(fabs(t)));
				}
			}
		}
			break;
		case Norm_L1: {
			for (int i = 0; i < mat.size(); ++i) {
				for (const auto& t : mat[i]) {
					*value += (double)(fabs(t));
				}
			}
		}
			break;
		case Norm_L2: {
			for (int i = 0; i < mat.size(); ++i) {
				for (const auto& t : mat[i]) {
					*value += t * t;
				}
			}
			*value = std::sqrt(*value);
		}
			break;
		default: {
			fprintf(stderr, "norm type is not supported\n");
			return -1;
		}
	}

	return 0;
}

int test_norm()
{
	fprintf(stderr, "test norm with C++:\n");
	 std::vector<int> norm_types{ 0, 1, 2 }; // positive infinity, L1, L2
	std::vector<std::string> str{ "Inf", "L1", "L2" };

	// 1. vector
	std::vector<float> vec1{ -2, 3, 1 };
	std::vector<std::vector<float>> tmp1(1);
	tmp1[0].resize(vec1.size());
	for (int i = 0; i < vec1.size(); ++i) {
		tmp1[0][i] = vec1[i];
	}

	for (int i = 0; i < str.size(); ++i) {
		double value{ 0.f };
		norm(tmp1, norm_types[i], &value);

		fprintf(stderr, "vector: %s: %f\n", str[i].c_str(), value);
	}

	// 2. matrix
	std::vector<float> vec2{ -3, 2, 0, 5, 6, 2, 7, 4, 8 };
	const int row_col{ 3 };
	std::vector<std::vector<float>> tmp2(row_col);
	for (int y = 0; y < row_col; ++y) {
		tmp2[y].resize(row_col);
		for (int x = 0; x < row_col; ++x) {
			tmp2[y][x] = vec2[y * row_col + x];
		}
	}

	for (int i = 0; i < str.size(); ++i) {
		double value{ 0.f };
		norm(tmp2, norm_types[i], &value);

		fprintf(stderr, "matrix: %s: %f\n", str[i].c_str(), value);
	}

	fprintf(stderr, "\ntest norm with opencv:\n");
	 norm_types[0] = 1; norm_types[1] = 2; norm_types[2] = 4; // positive infinity, L1, L2
	cv::Mat mat1(1, vec1.size(), CV_32FC1, vec1.data());

	for (int i = 0; i < norm_types.size(); ++i) {
		double value = cv::norm(mat1, norm_types[i]);
		fprintf(stderr, "vector: %s: %f\n", str[i].c_str(), value);
	}

	cv::Mat mat2(row_col, row_col, CV_32FC1, vec2.data());
	for (int i = 0; i < norm_types.size(); ++i) {
		double value = cv::norm(mat2, norm_types[i]);
		fprintf(stderr, "matrix: %s: %f\n", str[i].c_str(), value);
	}

	return 0;
}
