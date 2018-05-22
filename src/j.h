#ifndef __J_H__
#define __J_H__

#include "fdaPDE.h"
#include "mesh.h"
#include "regressionData.h"
#include "mixedFERegression.h"
#include "mesh_objects.h"

/**
 * @file j.h
 * @brief contains the class J
 */

/**
 * @class J
 * @brief Contains methods around the optimization of the functional J(f,K) as described in \cite Bernardi
 */
template <typename InputHandler, typename Integrator, UInt ORDER>
class J {
    protected:
        const MeshHandler<ORDER, 2, 2> & mesh_;
        const InputHandler & regressionData_;
        MixedFERegression<InputHandler, Integrator, ORDER, 2, 2> regression_;
        const std::vector<Point> & locations_;

    public:
		/**
		 * @detail The constructor solves the optimization of the functional J using the fdaPDE package
		 */
        J(const MeshHandler<ORDER, 2, 2> & mesh, const InputHandler & regressionData, const std::vector<Point> & locations);

		/**
		 * @return The solution coefficients of the FEM basis functions for fHat.
		 */
        const std::vector<VectorXr> & getSolution() const;

		/**
		 * @return The vector of DOF for each value of regressionData_.getLambda()
		 * @detail In the paper \cite Bernardi, DOF corresponds to tr(S) (p.14)
		 */
        const std::vector<Real> & getDOF() const;

		/**
		 * @return The vector of GCV indexes for each value of regressionData_.getLambda()
		 */
        VectorXr getGCV() const;
};

#include "j_imp.h"

#endif
