#include <iostream>
#include "Eigen/Eigen"
using namespace std;
using namespace Eigen;

void foo(MatrixXf& m)
{
    Matrix3f m2=Matrix3f::Zero(3,3);
    m2(0,0)=1;
    m=m2;
}  
int main()
{
    /* Definition, there is no initialization by default when defining, you must initialize yourself */
    MatrixXf m1(3,4);   //Dynamic matrix, build 3 rows and 4 columns.
    MatrixXf m2(4,3);   //4 rows and 3 columns, and so on.
    MatrixXf m3(3,3);
    Vector3f v1;        //If it is a static array, there is no need to specify the row or column
    /* Initialization */
    m1 = MatrixXf::Zero(3,4);       //Initialize with 0 matrix, specify the number of rows and columns
    m2 = MatrixXf::Zero(4,3);
    m3 = MatrixXf::Identity(3,3);   //Initialize with the identity matrix
    v1 = Vector3f::Zero();          //Similarly, if it is static, there is no need to specify the number of rows and columns

    m1 << 1,0,0,1,        // can also be initialized in this way
          1,5,0,1,
          0,0,9,1;
    m2 << 1,0,0,
          0,4,0,
          0,0,7,
          1,1,1;

    /* element access */
    v1[1] = 1;
    m3(2,2) = 7;
    cout<<"v1:\n"<<v1<<endl;
    cout<<"m3:\n"<<m3<<endl;
    /* Copy operation */
    VectorXf v2=v1;             //After copying, the number of rows and columns is equal to v1 on the right, and the matrix is ​​the same,
                                //You can also reset the number of rows and columns of the dynamic array in this way
    cout<<"v2:\n"<<v2<<endl;

    /* Matrix operation, you can achieve +-* / operation, you can also achieve continuous operation (but the number of dimensions must meet the situation),
         If m1, m2, m3 have the same dimensions, then m1 = m2 + m3 + m1; */
    m3 = m1 * m2;
    v2 += v1;
    cout<<"m3:\n"<<m3<<endl;
    cout<<"v2:\n"<<v2<<endl;
    //m3 = m3.transpose(); There is an error in this sentence, it is estimated that I cannot assign a value to myself
    cout<<"m3 transpose:\n"<<m3.transpose()<<endl;
    cout<<"m3 Determinant:\n"<<m3.determinant()<<endl;
    m3 = m3.inverse();
    cout<<"m3 inverse:\n"<<m3<<endl;

    system("pause");

    return 0;
}
