bool inverse_matrix(std::vector<std::vector<double> > & input, std::vector<std::vector<double> > & output)
{
    // TODO: Currently only supports 4-by-4 matrices, I can make this configurable.
    // see https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html

    Eigen::Matrix4d input_matrix;
    Eigen::Matrix4d output_matrix;
    Eigen::VectorXcd input_eivals;
    Eigen::VectorXcd output_eivals;

    input_matrix << input[0][0], input[0][1], input[0][2], input[0][3], 
                    input[1][0], input[1][1], input[1][2], input[1][3],
                    input[2][0], input[2][1], input[2][2], input[2][3],
                    input[3][0], input[3][1], input[3][2], input[3][3];
    cout << "Here is the matrix input:\n" << input_matrix << endl;

    input_eivals = input_matrix.eigenvalues();
    cout << "The eigenvalues of the input_eivals are:" << endl << input_eivals << endl;

    if(input_matrix.determinant() == 0) { return false; }
    output_matrix = input_matrix.inverse();
    cout << "Here is the matrix output:\n" << output_matrix << endl;

    output_eivals = output_matrix.eigenvalues();
    cout << "The eigenvalues of the output_eivals are:" << endl << output_eivals << endl;

    // Copy output_matrix to output
    for (int i = 0; i < 16; ++i)
    {
        int in = i/4;
        int im = i%4;
        output[in][im] = output_matrix(in, im);
    }
    return true;
}