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
template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
class HBase : public cppoptlib::BoundedProblem<Real, 2> {
    protected:
        const MeshHandler<ORDER, 2, 2> & mesh_;
        const InputHandler & regressionData_;
        const std::vector<Point> & locations_;

		/**
		 * @param x The anisotropy. x[0] is the angle, x[1] is the intensity of the anisotropy.
		 * @return The estimation of the spatial field at the locations, assumin an anisotropy x.
		 */
        const VectorXr fHat(const TVector & x) const;

    private:
        InputHandler createRegressionData(const TVector & x) const {
            return static_cast<const Derived*>(this)->createRegressionData(x);
        }

    public:
		/**
		 * @param mesh TODO
		 * @param regressionData
		 * @param locations
		 */
        using typename cppoptlib::BoundedProblem<Real, 2>::TVector;
        HBase(const MeshHandler<ORDER, 2, 2> & mesh, const InputHandler & regressionData, const std::vector<Point> & locations) : cppoptlib::BoundedProblem<Real, 2>(TVector(0., 1.), TVector(M_PI, 1000)), mesh_(mesh), regressionData_(regressionData), locations_(locations) {}

		/**
		 * @param x The anisotropy. x[0] is the angle, x[1] is the intensity of the anisotropy.
		 * @return The anisotropy in matrix form.
		 */
        static Eigen::Matrix<Real, 2, 2> buildKappa(const TVector & x);

		/**
		 * @param x The anisotropy. x[0] is the angle, x[1] is the intensity of the anisotropy.
		 * @return The value of the functional H(K). \cite Bernardi.
		 */
        Real value(const TVector & x);
};

template <typename InputHandler, typename Integrator, UInt ORDER>
struct H : public HBase<H<InputHandler, Integrator, ORDER>, InputHandler, Integrator, ORDER> {
        using typename HBase<H, InputHandler, Integrator, ORDER>::TVector;
        H(const MeshHandler<ORDER, 2, 2> & mesh, const InputHandler & regressionData, const std::vector<Point> & locations) : HBase<H, InputHandler, Integrator, ORDER>(mesh, regressionData, locations) {}

        Real value(const TVector & x) {
            REprintf("Option not implemented\n");
            exit(EXIT_FAILURE);
        }
};

#include "h_imp.h"

#endif
