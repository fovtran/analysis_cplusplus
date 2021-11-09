// test_trace
/*
Calculate the sum of diagonal elements of a two-dimensional array
*/

#include<iostream>
using namespace std;

double trMatrix( double** matrix, int n ){
    int tr;
    for( int i=0; i<n; i++ ){
        for( int j=0; j<n; j++){
            if( i==j ){
                                 tr + = * ((double *) matrix + i * n + j); // traverse a two-dimensional array with a pointer
            }
        }
    }
    return tr;
}

int main(){
    double matrix[3][3]={
        {2,2,3},
        {4,5,6},
        {7,8,1}
    };

    cout<<trMatrix((double**)matrix, 3);
}
