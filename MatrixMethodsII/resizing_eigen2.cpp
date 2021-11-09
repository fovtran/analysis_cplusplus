// resizing ii
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef Matrix<float, 1, 3> RVector;

int main()
{
	int cnt = 0;
	//Define a matrix and assign values
	MatrixXd m(2,2);
	m(0,0) = 3;
	m(1,0) = 2.5;
	m(0,1) = -1;
	m(1,1) = m(1,0) + m(0,1);
	cout << '[' << cnt++ << "] " << ": " << "m=" << endl;
	cout << m << endl;
	cout << "m.cols()=" << m.cols() << ", m.rows()=" << m.rows() << ", size()=" << m.size() << endl << endl;

	m << 1, 2, 3, 4;   //First row and then column
	cout << '[' << cnt++ << "] " << ": " << "Comma assignment, m=" << endl;
	cout << m << endl;
	cout << "The first line of m:" << m(1) << endl;

	//Generate a random matrix
	MatrixXd m1 = MatrixXd::Random(3,3);   //3 X 3 random matrix, the value is between -1 and 1
	cout << '[' << cnt++ << "] " << ": " << "m1 =" << endl << m1 << endl << endl;

	MatrixXd m2 = MatrixXd::Constant(3,3,1.2);  //3 x 3 matrix with value 1.2
	cout << '[' << cnt++ << "] " << ": " <<  "m2 = " << endl;
	cout << m2 << endl << endl;

	m1 = (m1 + m2) * 5;
	cout << '[' << cnt++ << "] " << ": " <<  "m1 = (m1 + m2) * 5 =" << endl << m1 << endl << endl;

	//Vector assignment and matrix multiplication---Comma-initialization
	VectorXd v(3);
	v << 1, 2, 3;
	cout << '[' << cnt++ << "] " << ": v" << endl << v << endl << endl;
	cout << '[' << cnt++ << "] " << "m1 * v =" << endl << m1 * v << endl << endl;

	//Assign value to vector through loop
	VectorXd v1(3);
	for(int i = 0; i < 3; i++)
		v1(i) = i;
	cout << '[' << cnt++ << "] " << "Column vector v1=" << endl << v1 << endl << endl;

	//Row vector
	RVector rv;
	for(int i = 0; i < 3; i++)
		rv(i) = i;
	cout << '[' << cnt++ << "] " << "Row vector:" << rv << endl << endl;

	//Resize the matrix
	MatrixXd m3 = MatrixXd::Random(3, 4);
	cout << '[' << cnt++ << "] " << ": " <<  "m3 = " << endl;
	cout << m3 << endl << endl;
	cout << "m3.resize(5, 5)=" << endl;
	m3.resize(5, 5);   //Do not keep the original value
	cout << m3 << endl << endl;

	MatrixXd m4= MatrixXd::Random(3, 3);
	cout << '[' << cnt++ << "] " << ": " <<  "m4 = " << endl;
	cout << m4 << endl;
	//Transpose of the matrix
	cout << "M4 transpose=" << endl;
	cout << m4.transpose() << endl;
	//The conjugate transpose of the matrix
	cout << "M4 Conjugation=" << endl;
	cout << m4.conjugate() << endl;
	//Adjoint transpose of matrix
	cout << "M4.adjoint=" << endl;
	cout << m4.adjoint() << endl << endl;

	//Matrix and scalar +, -, ×, / operation is omitted
	//Matrix and vector operations
	MatrixXd ma = MatrixXd::Random(2, 3);
	MatrixXd vb = MatrixXd::Random(3, 1);
	cout << '[' << cnt++ << "] " << ": " <<  "ma = " << endl;
	cout << ma << endl;
	cout << "vb = " << endl;
	cout << vb << endl;
	cout << "ma * vb = " << endl;
	cout << ma * vb << endl;

	//dot dot product and cross product slightly, see reference
	return 0;
}
