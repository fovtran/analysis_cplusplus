/**
    Cholesky decomposition implementation based on code by Gunter Winkler.
    Required for the Newton iteration in "example.cpp".

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#if !defined(__CHOLESKY_H)
#define __CHOLESKY_H

#include <cassert>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>

#include <boost/numeric/ublas/triangular.hpp>

/** \brief Compute the cholesky decomposition of a symmetric positive definite 
 * matrix A into LL^T. Returns L as a packed triangular matrix. When 
 * cholesky_decompose() returns false, a negative value was encountered 
 * inside a square root (A was not spd. or round-off errors caused the 
 * factorization to fail)
 */
template <typename MatType, typename TrType> bool cholesky_decompose(const MatType & A, TrType& L) {
	using namespace ublas;
	typedef typename MatType::value_type T;
	const size_t n = A.size1();
 
	assert(A.size1() == A.size2());
	assert(A.size1() == L.size1() 
		&& A.size2() == L.size2());

	for (size_t k=0 ; k<n; ++k) {
		const T temp = A(k,k) - inner_prod(project(row(L, k), range(0, k)),
						  project(row(L, k), range(0, k)));
		if (temp <= 0)
			return false;
		const T L_kk = std::sqrt(temp);
		L(k,k) = L_kk;

		matrix_column<TrType> col(L, k);
		project(col, range(k+1, n)) = (project(column(A, k), range(k+1, n))
			- prod(project(L, range(k+1, n), range(0, k)), 
			       project(row(L, k), range(0, k)))) / L_kk;
  }
  return true;      
}

/**
 * Solve a LL^Tx = b system.
 */
template <typename TrType, typename VecType> void cholesky_solve(const TrType &L, VecType& x) {
	using namespace ublas;

	inplace_solve(L, x, lower_tag());
	inplace_solve(trans(L), x, upper_tag());
}

#endif /* __CHOLESKY_H */
