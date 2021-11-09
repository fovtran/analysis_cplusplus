#include <xmmintrin.h>

inline Mat4* Mat4Mul(const Mat4 *M0, const Mat4 *M1, Mat4 *Out)
{
    Vec4 Col0 = {M1->M00, M1->M10, M1->M20, M1->M30};
    Vec4 Col1 = {M1->M01, M1->M11, M1->M21, M1->M31};
    Vec4 Col2 = {M1->M02, M1->M12, M1->M22, M1->M32};
    Vec4 Col3 = {M1->M03, M1->M13, M1->M23, M1->M33};

    Out->M00 = Vec4Dot(&M0->Row0, &Col0);
    Out->M01 = Vec4Dot(&M0->Row0, &Col1);
    Out->M02 = Vec4Dot(&M0->Row0, &Col2);
    Out->M03 = Vec4Dot(&M0->Row0, &Col3);

    Out->M10 = Vec4Dot(&M0->Row1, &Col0);
    Out->M11 = Vec4Dot(&M0->Row1, &Col1);
    Out->M12 = Vec4Dot(&M0->Row1, &Col2);
    Out->M13 = Vec4Dot(&M0->Row1, &Col3);

    Out->M20 = Vec4Dot(&M0->Row2, &Col0);
    Out->M21 = Vec4Dot(&M0->Row2, &Col1);
    Out->M22 = Vec4Dot(&M0->Row2, &Col2);
    Out->M23 = Vec4Dot(&M0->Row2, &Col3);

    Out->M30 = Vec4Dot(&M0->Row3, &Col0);
    Out->M31 = Vec4Dot(&M0->Row3, &Col1);
    Out->M32 = Vec4Dot(&M0->Row3, &Col2);
    Out->M33 = Vec4Dot(&M0->Row3, &Col3);

    return Out;
}


void dotFourByFourMatrix(const Mat4* left, const Mat4* right, Mat4* result) {
    const __m128 BCx = _mm_load_ps((float*)&B.Row0);
    const __m128 BCy = _mm_load_ps((float*)&B.Row1);
    const __m128 BCz = _mm_load_ps((float*)&B.Row2);
    const __m128 BCw = _mm_load_ps((float*)&B.Row3);

    float* leftRowPointer = &left->Row0;
    float* resultRowPointer = &result->Row0;

    for (unsigned int i = 0; i < 4; ++i, leftRowPointer += 4, resultRowPointer += 4) {
        __m128 ARx = _mm_set1_ps(leftRowPointer[0]);
        __m128 ARy = _mm_set1_ps(leftRowPointer[1]);
        __m128 ARz = _mm_set1_ps(leftRowPointer[2]);
        __m128 ARw = _mm_set1_ps(leftRowPointer[3]);

        __m128 X = ARx * BCx;
        __m128 Y = ARy * BCy;
        __m128 Z = ARz * BCz;
        __m128 W = ARw * BCw;

        __m128 R = X + Y + Z + W;
        _mm_store_ps(resultRowPointer, R);
    }
}

__m128 BCx = _mm_load_ps((float*)&B.Row0);
__m128 BCy = _mm_load_ps((float*)&B.Row1);
__m128 BCz = _mm_load_ps((float*)&B.Row2);
__m128 BCw = _mm_load_ps((float*)&B.Row3);


// Calculate Row0 in resulting matrix
__m128 ARx = _mm_set1_ps(A.Row0.X);
__m128 ARy = _mm_set1_ps(A.Row0.Y);
__m128 ARz = _mm_set1_ps(A.Row0.Z);
__m128 ARw = _mm_set1_ps(A.Row0.W);

__m128 X = _mm_mul_ps(ARx, BCx);
__m128 Y = _mm_mul_ps(ARy, BCy);
__m128 Z = _mm_mul_ps(ARz, BCz);
__m128 W = _mm_mul_ps(ARw, BCw);

__m128 R = _mm_add_ps(X, _mm_add_ps(Y, _mm_add_ps(Z, W)));
_mm_storeu_ps((float*)&Result.Row0, R);

// Calculate Row1 in resulting matrix
ARx = _mm_set1_ps(A.Row1.X);
ARy = _mm_set1_ps(A.Row1.Y);
ARz = _mm_set1_ps(A.Row1.Z);
ARw = _mm_set1_ps(A.Row1.W);

X = _mm_mul_ps(ARx, BCx);
Y = _mm_mul_ps(ARy, BCy);
Z = _mm_mul_ps(ARz, BCz);
W = _mm_mul_ps(ARw, BCw);

R = _mm_add_ps(X, _mm_add_ps(Y, _mm_add_ps(Z, W)));
_mm_storeu_ps((float*)&Result.Row1, R);

// Calculate Row2 in resulting matrix
ARx = _mm_set1_ps(A.Row2.X);
ARy = _mm_set1_ps(A.Row2.Y);
ARz = _mm_set1_ps(A.Row2.Z);
ARw = _mm_set1_ps(A.Row2.W);

X = _mm_mul_ps(ARx, BCx);
Y = _mm_mul_ps(ARy, BCy);
Z = _mm_mul_ps(ARz, BCz);
W = _mm_mul_ps(ARw, BCw);

R = _mm_add_ps(X, _mm_add_ps(Y, _mm_add_ps(Z, W)));
_mm_storeu_ps((float*)&Result.Row2, R);

// Calculate Row3 in resulting matrix
ARx = _mm_set1_ps(A.Row3.X);
ARy = _mm_set1_ps(A.Row3.Y);
ARz = _mm_set1_ps(A.Row3.Z);
ARw = _mm_set1_ps(A.Row3.W);

X = _mm_mul_ps(ARx, BCx);
Y = _mm_mul_ps(ARy, BCy);
Z = _mm_mul_ps(ARz, BCz);
W = _mm_mul_ps(ARw, BCw);

R = _mm_add_ps(X, _mm_add_ps(Y, _mm_add_ps(Z, W)));
_mm_storeu_ps((float*)&Result.Row3, R);
