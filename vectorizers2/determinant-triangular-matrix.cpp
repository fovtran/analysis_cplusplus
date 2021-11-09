#include <iostream>
#include <Eigen>
#include <fstream>
#include <chrono>
using namespace Eigen;
using namespace std;
using namespace std::chrono;
ifstream fin("input.txt");

int main()
{
    double aux;
    int n = 10;
    MatrixXd A;
    A.resize(n,n);

    // Read A
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++){
            fin>>aux;
            A(i,j) = aux;
        }
    cout<<"Start!"<<endl;
    cout<<A.determinant()<<endl;


    //Use QR decomposition, get R matrix
    HouseholderQR<MatrixXd> qr(A);
    qr.compute(A);
    MatrixXd R = qr.matrixQR().template  triangularView<Upper>();

    // R is a triangular matrix, det(A) should be equal to det(R)
    cout<<R.determinant()<<endl;

    return 0;
}