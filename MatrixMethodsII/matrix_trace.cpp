// trace of MatrixXi


#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "common.hpp"

// ================================ Find the trace of the matrix =========== ======================
template<typename _Tp>
_Tp trace(const std::vector<std::vector<_Tp>>& mat)
{
	_Tp ret{ (_Tp)0 };
	int nm = std::min(mat.size(), mat[0].size());

	for (int i = 0; i < nm; ++i) {
		ret += mat[i][i];
	}

	return ret;
}

int test_trace()
{
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
					{ -3.6f, 9.2f, 0.5f, 7.2f },
					{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	fprintf(stderr, "source matrix:\n");
	print_matrix(vec);

	float tr = trace(vec);
	fprintf(stderr, "\nc++ implement trace: %f\n", tr);

	cv::Mat mat(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			mat.at<float>(y, x) = vec.at(y).at(x);
		}
	}

	cv::Scalar scalar = cv::trace(mat);
	fprintf(stderr, "\nopencv implement trace: %f\n", scalar.val[0]);

	return 0;
}
