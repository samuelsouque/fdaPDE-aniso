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
        const std::vector<Point> & meshLoc_;
        const InputHandler & regressionData_;
        MixedFERegression<InputHandler, Integrator, ORDER, 2, 2> regression_;

    public:
		/**
		 * @detail The constructor solves the optimization of the functional J using the fdaPDE package
		 */
        J(const MeshHandler<ORDER, 2, 2> & mesh, const std::vector<Point> & meshLoc, const InputHandler & regressionData);
		
        /**
		 * @return The vector of GCV indexes for each value of regressionData_.getLambda()
		 */
        VectorXr getGCV() const;
};

#include "j_imp.h"

#endif
