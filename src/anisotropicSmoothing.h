#ifndef __ANISOTROPIC_SMOOTHING_H__
#define __ANISOTROPIC_SMOOTHING_H__

#include "fdaPDE.h"
#include "mesh.h"
#include "regressionData.h"
#include "h.h"


/**
 * @file anisotropicSmoothing.h
 * @brief contains the class template AnisotropicSmoothingBase implementing the main algorithm
 */

/**
 * @class AnisotropicSmoothing
 * @brief Implements the anisotropic smoothing algorithm, with estimation of the anisotropy matrix from data
 */
template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
class AnisotropicSmoothingBase {
    protected:
        using TVector = typename H<InputHandler, Integrator, ORDER>::TVector;
        const MeshHandler<ORDER, 2, 2> & mesh_;
        const InputHandler & regressionData_;
        const std::vector<Real> & lambda_;
        const std::vector<Point> & locations_;

    private:

		/**
		 * @brief creates a new RegressionData object from regressionData_, forcing the regularization coefficient to lambda
		 */
        InputHandler createRegressionData(const Real & lambda) const {
            return static_cast<const Derived*>(this)->createRegressionData(lambda);
        }

		/**
		 * @brief creates a new RegressionData object from regressionData_, forcing the anisotropy to k and DOF_ to dof
		 */
        InputHandler createRegressionData(const TVector & k, const bool dof) const {
            return static_cast<const Derived*>(this)->createRegressionData(k, dof);
        }

		/**
		 * @brief creates a new RegressionData object from regressionData_, forcing the regularization coefficient to lambda and the anisotropy to k
		 */
        InputHandler createRegressionData(const Real & lambda, const TVector & k) const {
            return static_cast<const Derived*>(this)->createRegressionData(lambda, k);
        }

    public:
        AnisotropicSmoothingBase(const MeshHandler<ORDER, 2, 2> & mesh, const InputHandler & regressionData, const std::vector<Real> & lambda, const std::vector<Point> & locations) : mesh_(mesh), regressionData_(regressionData), lambda_(lambda), locations_(locations) {}

		/**
		 * @brief executes the anisotropic smoothing algorithm for the problem described in the class attributes
		 * @return the first element of the pair is the vector of solution coefficients, the second element is the estimated anisotropy matrix
		 */
        std::pair<const std::vector<VectorXr>, const TVector> smooth() const;
};

/**
 * @class AnisotropicSmoothing
 * @brief Non specialized template class to handle an unimplemented InputHandler argument. The template must be specialized otherwise.
 */
template <typename InputHandler, typename Integrator, UInt ORDER>
struct AnisotropicSmoothing : public AnisotropicSmoothingBase<AnisotropicSmoothing<InputHandler, Integrator, ORDER>, InputHandler, Integrator, ORDER> {
        using TVector = typename AnisotropicSmoothingBase<AnisotropicSmoothing, InputHandler, Integrator, ORDER>::TVector;
        AnisotropicSmoothing(const MeshHandler<ORDER, 2, 2> & mesh, const InputHandler & regressionData, const std::vector<Real> & lambda, const std::vector<Point> & locations) : AnisotropicSmoothingBase<AnisotropicSmoothing, InputHandler, Integrator, ORDER>(mesh, regressionData, lambda, locations) {}
		
        std::pair<const std::vector<VectorXr>, const TVector> smooth() const {
            REprintf("Anisotropic smoothing not implemented for such an input handler\n");
            return std::pair<const std::vector<VectorXr>, const TVector>();
        }
};

#include "anisotropicSmoothing_imp.h"

#endif
