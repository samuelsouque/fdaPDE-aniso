#ifndef __H_H__
#define __H_H__

#include "fdaPDE.h"
#include "boundedproblem.h"
#include "mesh.h"
#include "regressionData.h"
#include "mesh_objects.h"

/**
 *	@file h.h
 *	@brief contains the class declaration of the functional H(K).
 */

/**
 * 	@class H
 *	@brief Implements the functional H(K) as described in the paper \cite Bernardi.
 *	@details We use the cppoptlib library for its LBFGSB algorithm.
 *		The class H inheritates from a cppoptlib class for compatibility reasons with cppoptlib.
 */
template <typename InputHandler, typename Integrator, UInt ORDER>
class H : public cppoptlib::BoundedProblem<Real, 2> {
    private:
        const MeshHandler<ORDER, 2, 2> &mesh_;
        const std::vector<Point> &meshLoc_;
        InputHandler &regressionData_;

		/**
		 * @param anisoParam The anisotropy. anisoParam[0] is the angle, anisoParam[1] is the intensity of the anisotropy.
		 * @return The estimation of the spatial field at the locations, assumin an anisotropy anisoParam.
		 */
        const VectorXr fHat(const TVector &anisoParam) const;

    public:
		/**
		 * @param mesh TODO
		 * @param regressionData
		 */
        using typename cppoptlib::BoundedProblem<Real, 2>::TVector;
        H(const MeshHandler<ORDER, 2, 2> &mesh, const std::vector<Point> &meshLoc, InputHandler &regressionData) : cppoptlib::BoundedProblem<Real, 2>(TVector(0., 1.), TVector(M_PI, 1000.)), mesh_(mesh), meshLoc_(meshLoc), regressionData_(regressionData) {}

		/**
		 * @param anisoParam The anisotropy. anisoParam[0] is the angle, anisoParam[1] is the intensity of the anisotropy.
		 * @return The anisotropy in matrix form.
		 */
        static Eigen::Matrix<Real, 2, 2> buildKappa(const TVector &anisoParam);

		/**
		 * @param anisoParam The anisotropy. anisoParam[0] is the angle, anisoParam[1] is the intensity of the anisotropy.
		 * @return The value of the functional H(K). \cite Bernardi.
		 */
        Real value(const TVector &anisoParam);
        void gradient(const TVector &anisoParam, TVector &grad) {
            finiteGradient(anisoParam, grad, 0);
        }
};

#include "h_imp.h"

#endif
