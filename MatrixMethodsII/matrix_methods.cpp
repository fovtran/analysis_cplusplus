#include<Eigen/Dense>
...
//1D objects
Vector4d  v4;
Vector2f  v1(x, y);
Array3i   v2(x, y, z);
Vector4d  v3(x, y, z, w);
VectorXf  v5; // empty object
ArrayXf   v6(size);

//2D objects
atrix4f  m1;
MatrixXf  m5; // empty object
MatrixXf  m6(nb_rows, nb_columns);


Vector3f  v1;     v1 << x, y, z;
ArrayXf   v2(4);  v2 << 1, 2, 3, 4;
Matrix3f  m1;   m1 << 1, 2, 3,
                      4, 5, 6,
                      7, 8, 9;


                      int rows=5, cols=5;
                      MatrixXf m(rows,cols);
                      m << (Matrix3f() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished(),
                           MatrixXf::Zero(3,cols-3),
                           MatrixXf::Zero(rows-3,3),
                           MatrixXf::Identity(rows-3,cols-3);
                      cout << m;
