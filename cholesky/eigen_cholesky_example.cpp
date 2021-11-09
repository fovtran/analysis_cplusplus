/*  This is an example on how automatic differentiation can be used in
    a multidimensional Newton iteration, which eliminates need for
    (often painful) symbolic derivatives.

    Copyright (c) 2009 by Wenzel Jakob

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

#include <iostream>
#include <Eigen/Cholesky>
#include "autodiff.h"

DECLARE_DIFFSCALAR_BASE();

int main(int argc, char **argv) {
	typedef Eigen::Vector2d                     Gradient;
	typedef Eigen::Matrix2d                     Hessian;
	typedef DScalar2<double, Gradient, Hessian> DScalar;

	/* Do a multidimensional newton iteration for the function
	 		f(x0, x1) = -log(1-x0-x1) - log(x0) - log(x1)

	   starting at x0 = .85 and x1 = .05
	*/
	Eigen::Vector2d x(0.85, 0.05);

	/* There are two independent variables */
	DiffScalarBase::setVariableCount(2);

	for (int i=0; i<10; ++i) {
		/* Print the current iterate -- it should end
		   up converging to [1/3, 1/3] */
		std::cout << "Iteration " << i << "  (x=" << x.transpose() << ")" << std::endl;

		/* Get a reference to the independent variables. These are basically
		   large data structures, which contain the value, gradient and
		   hessian of the two functions
		   	f(x0,x1) = x0   (=> gradient=[1; 0], hessian = [0 0; 0 0])
			f(x0,x1) = x1   (=> gradient=[0; 1], hessian = [0 0; 0 0])
		   evaluated at the current x0,x1 values.
		*/
		DScalar x0(0, x[0]), x1(1, x[1]);

		/* Compute f(x1, x1). This additionally propagates first and
		   second derivative information through builtin operators and
		   math library calls.
		*/
		DScalar Fx = -log(1-x0-x1) - log(x0) - log(x1);

		/* Print the local derivative information */
		std::cout << "  - Local information: " << Fx << std::endl << std::endl;

		/* Compute the Cholesky decomposition of f's Hessian
		   and use it to take a full Newton step */
		x -= Fx.getHessian().llt().solve(Fx.getGradient());
	}

	return 0;
}
