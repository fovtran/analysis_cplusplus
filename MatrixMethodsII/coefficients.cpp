// coefficients of Matrix
#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
int main()
{
    MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    cout << "Here is the matrix m:" << endl;
    cout << m << endl;

    VectorXd v(2);
    v(0) = 4;
    v[1] = v[0] - 1;     //operator[] is overloaded in vectors and has the same meaning as ()
    cout << "Here is the vector v:" << endl;
    cout << v << endl;

    getchar();
    getchar();
}
