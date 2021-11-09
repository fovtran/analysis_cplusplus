// eigenvals and eivengevot
#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "common.hpp"

template<typename _Tp>
static inline _Tp hypot(_Tp a, _Tp b)
{
    a = std::abs(a);
    b = std::abs(b);
    if (a > b) {  
        b /= a;
        return a*std::sqrt(1 + b*b);
    }
    if (b > 0) {
        a /= b;
        return b*std::sqrt(1 + a*a);
    }
    return 0;
}

template<typename _Tp>
int eigen(const std::vector<std::vector<_Tp>>& mat, std::vector<_Tp>& eigenvalues, std::vector<std::vector<_Tp>>& eigenvectors, bool sort_ = true)
{
    auto n = mat.size();
    for (const auto& m : mat) {
        if (m.size() != n) {
            fprintf(stderr, "mat must be square and it should be a real symmetric matrix\n");
            return -1;
        }
    }

    eigenvalues.resize(n, (_Tp)0);
    std::vector<_Tp> V(n*n, (_Tp)0);
    for (int i = 0; i < n; ++i) {
        V[n * i + i] = (_Tp)1;
        eigenvalues[i] = mat[i][i];
    }

    const _Tp eps = std::numeric_limits<_Tp>::epsilon();
    int maxIters{ (int)n * (int)n * 30 };
    _Tp mv{ (_Tp)0 };
    std::vector<int> indR(n, 0), indC(n, 0);
    std::vector<_Tp> A;
    for (int i = 0; i < n; ++i) {
        A.insert(A.begin() + i * n, mat[i].begin(), mat[i].end());
    }

    for (int k = 0; k < n; ++k) {
        int m, i;
        if (k < n - 1) {
            for (m = k + 1, mv = std::abs(A[n*k + m]), i = k + 2; i < n; i++) {
                _Tp val = std::abs(A[n*k + i]);
                if (mv < val)
                    mv = val, m = i;
            }
            indR[k] = m;
        }
        if (k > 0) {
            for (m = 0, mv = std::abs(A[k]), i = 1; i < k; i++) {
                _Tp val = std::abs(A[n*i + k]);
                if (mv < val)
                    mv = val, m = i;
            }
            indC[k] = m;
        }
    }

    if (n > 1) for (int iters = 0; iters < maxIters; iters++) {
        int k, i, m;
        // find index (k,l) of pivot p
        for (k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n - 1; i++) {
            _Tp val = std::abs(A[n*i + indR[i]]);
            if (mv < val)
                mv = val, k = i;
        }
        int l = indR[k];
        for (i = 1; i < n; i++) {
            _Tp val = std::abs(A[n*indC[i] + i]);
            if (mv < val)
                mv = val, k = indC[i], l = i;
        }

        _Tp p = A[n*k + l];
        if (std::abs(p) <= eps)
            break;
        _Tp y = (_Tp)((eigenvalues[l] - eigenvalues[k])*0.5);
        _Tp t = std::abs(y) + hypot(p, y);
        _Tp s = hypot(p, t);
        _Tp c = t / s;
        s = p / s; t = (p / t)*p;
        if (y < 0)
            s = -s, t = -t;
        A[n*k + l] = 0;

        eigenvalues[k] -= t;
        eigenvalues[l] += t;

        _Tp a0, b0;

#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c

        // rotate rows and columns k and l
        for (i = 0; i < k; i++)
            rotate(A[n*i + k], A[n*i + l]);
        for (i = k + 1; i < l; i++)
            rotate(A[n*k + i], A[n*i + l]);
        for (i = l + 1; i < n; i++)
            rotate(A[n*k + i], A[n*l + i]);

        // rotate eigenvectors
        for (i = 0; i < n; i++)
            rotate(V[n*k+i], V[n*l+i]);

#undef rotate

        for (int j = 0; j < 2; j++) {
            int idx = j == 0 ? k : l;
            if (idx < n - 1) {
                for (m = idx + 1, mv = std::abs(A[n*idx + m]), i = idx + 2; i < n; i++) {
                    _Tp val = std::abs(A[n*idx + i]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indR[idx] = m;
            }
            if (idx > 0) {
                for (m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++) {
                    _Tp val = std::abs(A[n*i + idx]);
                    if (mv < val)
                        mv = val, m = i;
                }
                indC[idx] = m;
            }
        }
    }

    // sort eigenvalues & eigenvectors
    if (sort_) {
        for (int k = 0; k < n - 1; k++) {
            int m = k;
            for (int i = k + 1; i < n; i++) {
                if (eigenvalues[m] < eigenvalues[i])
                    m = i;
            }
            if (k != m) {
                std::swap(eigenvalues[m], eigenvalues[k]);
                for (int i = 0; i < n; i++)
                    std::swap(V[n*m+i], V[n*k+i]);
            }
        }
    }

    eigenvectors.resize(n);
    for (int i = 0; i < n; ++i) {
        eigenvectors[i].resize(n);
        eigenvectors[i].assign(V.begin() + i * n, V.begin() + i * n + n);
    }

    return 0;
}

int test_eigenvalues_eigenvectors()
{
    std::vector<float> vec{ 1.23f, 2.12f, -4.2f,
        2.12f, -5.6f, 8.79f,
        -4.2f, 8.79f, 7.3f };
    const int N{ 3 };

    fprintf(stderr, "source matrix:\n");
    int count{ 0 };
    for (const auto& value : vec) {
        if (count++ % N == 0) fprintf(stderr, "\n");
        fprintf(stderr, "  %f  ", value);
    }
    fprintf(stderr, "\n\n");

    fprintf(stderr, "c++ compute eigenvalues and eigenvectors, sort:\n");
    std::vector<std::vector<float>> eigen_vectors1, mat1;
    std::vector<float> eigen_values1;
    mat1.resize(N);
    for (int i = 0; i < N; ++i) {
        mat1[i].resize(N);
        for (int j = 0; j < N; ++j) {
            mat1[i][j] = vec[i * N + j];
        }
    }

    if (eigen(mat1, eigen_values1, eigen_vectors1, true) != 0) {
        fprintf(stderr, "campute eigenvalues and eigenvector fail\n");
        return -1;
    }

    fprintf(stderr, "eigenvalues:\n");
    std::vector<std::vector<float>> tmp(N);
    for (int i = 0; i < N; ++i) {
        tmp[i].resize(1);
        tmp[i][0] = eigen_values1[i];
    }
    print_matrix(tmp);

    fprintf(stderr, "eigenvectors:\n");
    print_matrix(eigen_vectors1);

    fprintf(stderr, "c++ compute eigenvalues and eigenvectors, no sort:\n");
    if (eigen(mat1, eigen_values1, eigen_vectors1, false) != 0) {
        fprintf(stderr, "campute eigenvalues and eigenvector fail\n");
        return -1;
    }

    fprintf(stderr, "eigenvalues:\n");
    for (int i = 0; i < N; ++i) {
        tmp[i][0] = eigen_values1[i];
    }
    print_matrix(tmp);

    fprintf(stderr, "eigenvectors:\n");
    print_matrix(eigen_vectors1);

    fprintf(stderr, "\nopencv compute eigenvalues and eigenvectors:\n");
    cv::Mat mat2(N, N, CV_32FC1, vec.data());

    cv::Mat eigen_values2, eigen_vectors2;
    bool ret = cv::eigen(mat2, eigen_values2, eigen_vectors2);
    if (!ret) {
        fprintf(stderr, "fail to run cv::eigen\n");
        return -1;
    }

    fprintf(stderr, "eigenvalues:\n");
    print_matrix(eigen_values2);

    fprintf(stderr, "eigenvectors:\n");
    print_matrix(eigen_vectors2);

    return 0;
}
