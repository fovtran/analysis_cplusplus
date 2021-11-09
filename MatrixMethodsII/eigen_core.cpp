#include <iostream>
#include "Eigen\Core"

//import most common Eigen types
using namespace Eigen;

int main()
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);

    std::cout<<"Hear is the matrix m:\n"<<m<<std::endl;
    VectorXd v(2);
    v(0) = 4;
    v(1) = v(0) - 1;
    std::cout<<"Here is the vector v:\n"<<v<<std::endl;
}

#include <iostream>
#include "Eigen\Core"

using namespace Eigen;

int main()
{
    MatrixXd m(2,5);
    m<<1,2,3,4,5,
       6,7,8,9,10;
    m.resize(4,3);
    std::cout<<"The matrix m is:\n"<<m<<std::endl;
    std::cout<<"The matrix m is of size "
             <<m.rows()<<"x"<<m.cols()<<std::endl;
    std::cout<<"It has "<<m.size()<<" coefficients"<<std::endl;
    VectorXd v(2);
    v<<1,2;
    v.resize(5);
    std::cout<<"The vector v is:\n"<<v<<std::endl;
    std::cout<<"The vector v is of size "<<v.size()<<std::endl;
    std::cout<<"As a matrix,v is of size "<<v.rows()
             <<"x"<<v.cols()<<std::endl;
}

using namespace Eigen;

int main()
{
    MatrixXf a(2,2);
    MatrixXf b(3,3);
    b<<1,2,3,
       4,5,6,
       7,8,9;
    a = b;
    std::cout<<a<<std::endl;
}
