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
template <typename InputHandler, typename Integrator, UInt ORDER>
class AnisotropicSmoothing {
    private:
        using TVector = typename H<InputHandler, Integrator, ORDER>::TVector;

        InputHandler &regressionData_;
        const MeshHandler<ORDER, 2, 2> &mesh_;
        const std::vector<Point> meshLoc_;
        const std::vector<Real> lambda_;
        const std::vector<Real> lambdaCrossVal_;
        bool dof_;

        std::vector<Real> seq(const UInt &n_obs, const Real &area) const;

    public:
        AnisotropicSmoothing(InputHandler &regressionData, const MeshHandler<ORDER, 2, 2> &mesh) : regressionData_(regressionData), mesh_(mesh), meshLoc_(computeMeshLoc()), lambda_(regressionData_.getLambda()), lambdaCrossVal_(seq(regressionData.getNumberofObservations(), mesh.getArea())), dof_(regressionData_.computeDOF()) {}

        std::vector<Point> computeMeshLoc() {
            std::vector<Point> locations;
            if (regressionData_.isLocationsByNodes()) {
                const std::vector<UInt> &observationsIndices = regressionData_.getObservationsIndices();
                locations.reserve(observationsIndices.size());
                for (std::vector<UInt>::size_type i = 0U; i < observationsIndices.size(); i++) {
                    Id id = observationsIndices[i];
                    Point point = mesh_.getPoint(id);
                    locations.push_back(point);
                }
            }
            return locations;
        }

		/**
		 * @brief executes the anisotropic smoothing algorithm for the problem described in the class attributes
		 * @return the first element of the pair is the vector of solution coefficients, the second element is the estimated anisotropy matrix
		 */
        std::pair<const std::vector<VectorXr>, const TVector> smooth() const;
};

#include "anisotropicSmoothing_imp.h"

#endif
