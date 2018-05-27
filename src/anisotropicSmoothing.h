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
        static constexpr std::array<Real, 70> lambdaCrossVal_ = {1.000000e-07, 1.258925e-07, 1.584893e-07, 1.995262e-07, 2.511886e-07, 3.162278e-07, 3.981072e-07, 5.011872e-07, 6.309573e-07, 7.943282e-07, 1.000000e-06, 1.258925e-06, 1.584893e-06, 1.995262e-06, 2.511886e-06, 3.162278e-06, 3.981072e-06, 5.011872e-06, 6.309573e-06, 7.943282e-06, 1.000000e-05, 1.258925e-05, 1.584893e-05, 1.995262e-05, 2.511886e-05, 3.162278e-05, 3.981072e-05, 5.011872e-05, 6.309573e-05, 7.943282e-05, 1.000000e-04, 1.258925e-04, 1.584893e-04, 1.995262e-04, 2.511886e-04, 3.162278e-04, 3.981072e-04, 5.011872e-04, 6.309573e-04, 7.943282e-04, 1.000000e-03, 1.258925e-03, 1.584893e-03, 1.995262e-03, 2.511886e-03, 3.162278e-03, 3.981072e-03, 5.011872e-03, 6.309573e-03, 7.943282e-03, 1.000000e-02, 1.258925e-02, 1.584893e-02, 1.995262e-02, 2.511886e-02, 3.162278e-02, 3.981072e-02, 5.011872e-02, 6.309573e-02, 7.943282e-02, 1.000000e-01, 1.258925e-01, 1.584893e-01, 1.995262e-01, 2.511886e-01, 3.162278e-01, 3.981072e-01, 5.011872e-01, 6.309573e-01, 7.943282e-01};
        using TVector = typename H<InputHandler, Integrator, ORDER>::TVector;
        const MeshHandler<ORDER, 2, 2> & mesh_;
        const InputHandler & regressionData_;

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
        AnisotropicSmoothingBase(const InputHandler & regressionData, const MeshHandler<ORDER, 2, 2> & mesh) : regressionData_(regressionData), mesh_(mesh) {}

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

        AnisotropicSmoothing(const InputHandler & regressionData, const MeshHandler<ORDER, 2, 2> & mesh) : AnisotropicSmoothingBase<AnisotropicSmoothing, InputHandler, Integrator, ORDER>(regressionData, mesh) {}
		
        std::pair<const std::vector<VectorXr>, const TVector> smooth() const {
            REprintf("Anisotropic smoothing not implemented for such an input handler\n");
            return std::pair<const std::vector<VectorXr>, const TVector>();
        }
};

#include "anisotropicSmoothing_imp.h"

#endif
