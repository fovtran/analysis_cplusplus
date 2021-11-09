int test_mat_determinant()
{
	std::vector<float> vec{ 1, 0, 2, -1, 3, 0, 0, 5, 2, 1, 4, -3, 1, 0, 5, 0 };

	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(vec.data(), 4, 4);
	double det = map.determinant();
	fprintf(stderr, "det: %f\n", det);

	return 0;
}
