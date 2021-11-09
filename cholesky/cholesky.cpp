/*  Copyright (c) 2009 by Wenzel Jakob */

#include <iostream>
#include "autodiff.h"
#include "cholesky.h"

using namespace std;

int main(int argc, char **argv) {
	/* Instantiate a differentiation environment with 2 independent variables.
	   (this is placed on the heap - alternatively, the typedef
	    could be changed to
			typedef DiffEnvFixed<double, 2> DEnv;
		in which case the variable count is fixed at compile time and the
		DEnv instance is placed on the stack instead)
	*/

	typedef DiffEnv<double> DEnv;
	DEnv env(2);

	/* Do a multidimensional newton iteration for the function
	 		f(x1, x2) = -log(1-x1-x2) - log(x1) - log(x2)

	   starting at x1 = .85 and x2 = .05
	*/
	env[0] = 0.85; env[1] = 0.05;

	for (int i=0; i<10; ++i) {
		/* Print the current iterate -- it should end
		   up converging to [1/3, 1/3]
		 */
		cout << "Iteration " << i << "  (x1=" << env[0] << ", x2="
			 << env[1] << ")" << endl;

		/* Get a reference to the independent variables. These are basically
		   large data structures, which contain the value, gradient and
		   hessian of the two functions
		   	f(x1,x2) = x1   (=> gradient=[1; 0], hessian = [0 0; 0 0])
			f(x1,x2) = x2   (=> gradient=[0; 1], hessian = [0 0; 0 0])
		   evaluated at the current x1,x2 values.
		*/
		DEnv::Scalar x1 = env.getIndepScalar(0);
		DEnv::Scalar x2 = env.getIndepScalar(1);

		/* Compute f(x1, x2). This additionally propagates first and
		   second derivative information through builtin operators and
		   math library calls.
		*/
		DEnv::Scalar Fx = -log(1-x1-x2) - log(x1) - log(x2);

		/* Print the local derivative information */
		cout << "  - Local information: " << Fx << endl << endl;

		/* Compute the Cholesky decomposition of f's Hessian
		   and use it to take a full Newton step */
		DEnv::Hessian hess  = Fx.getHessian();
		DEnv::Gradient grad = Fx.getGradient();
		ublas::triangular_matrix<double, ublas::lower, ublas::row_major> tri(2, 2);

		if (!cholesky_decompose(hess, tri))
			throw std::range_error("Matrix is not symmetric positive definite!");

		cholesky_solve(tri, grad);

		/* Update the differentiation environment with new coefficients */
		env[0] -= grad(0);
		env[1] -= grad(1);
	}

	return 0;
}
