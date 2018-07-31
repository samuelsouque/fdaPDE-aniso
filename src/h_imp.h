#ifndef __H_IMP_H__
#define __H_IMP_H__

#include "evaluatorExt.h"
#include "mixedFERegression.h"
#include "mesh_objects.h"

#include <chrono>

template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
Real HBase<Derived, InputHandler, Integrator, ORDER>::value(const TVector & anisoParam) {
    /*Calculating fHat*/
    VectorXr estimations = fHat(anisoParam);

    /* Calculating H */
    estimations -= regressionData_.getObservations();
    Real H = estimations.squaredNorm();

    return H;
}

template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
const VectorXr HBase<Derived, InputHandler, Integrator, ORDER>::fHat(const TVector & anisoParam) const {
    ///const InputHandler & data = createRegressionData(anisoParam);
    regressionData_.setK(buildKappa(anisoParam));
    // Compute the regression coefficients
    MixedFERegression<RegressionDataElliptic, Integrator, ORDER, 2, 2> regression(mesh_, regressionData_);
    regression.apply();
    const std::vector<VectorXr> & solution = regression.getSolution();
    
    /*std::vector<Point> locations;
    if (regressionData_.isLocationsByNodes()) {
        const std::vector<UInt> & observationsIndices = regressionData_.getObservationsIndices();
        locations.reserve(observationsIndices.size());
        for (std::vector<UInt>::size_type i = 0U; i < observationsIndices.size(); i++) {
            Id id = observationsIndices[i];
            Point point = mesh_.getPoint(id);
            locations.push_back(point);
        }
    } else {
        locations = regressionData_.getLocations();
    }*/

    // Return the evaluation of the coefficients at the data points
    EvaluatorExt<ORDER> evaluator(mesh_);
    const VectorXr result = evaluator.eval(solution, regressionData_.isLocationsByNodes() ? meshLoc_ : regressionData_.getLocations());

    return result;
}

template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
Eigen::Matrix<Real, 2, 2> HBase<Derived, InputHandler, Integrator, ORDER>::buildKappa(const TVector & anisoParam) {
    // Building K from anisoParam as an Eigen matrix
    // anisoParam[0] is the angle α (alpha)
    // anisoParam[1] is the intensity γ (gamma)

    Eigen::Matrix<Real,2,2> Q;
    Q << std::cos(anisoParam(0)), -std::sin(anisoParam(0)), 
          std::sin(anisoParam(0)), std::cos(anisoParam(0));

    Eigen::Matrix<Real,2,2> Sigma;
    // Deal with invalid intensity argument
    Real intensity = std::abs(anisoParam(1));
    Sigma << 1/std::sqrt(intensity), 0., 
          0., intensity/std::sqrt(intensity);

    // Kappa = Q * Sigma * Q.inverse()
    Eigen::Matrix<Real, 2, 2> Kappa = Q * Sigma * Q.inverse();
    return Kappa;
}

template <typename Integrator, UInt ORDER>
struct H<RegressionDataElliptic, Integrator, ORDER> : public HBase<H<RegressionDataElliptic, Integrator, ORDER>, RegressionDataElliptic, Integrator, ORDER> {
        using typename HBase<H<RegressionDataElliptic, Integrator, ORDER>, RegressionDataElliptic, Integrator, ORDER>::TVector;
        H(const MeshHandler<ORDER, 2, 2> & mesh, const std::vector<Point> & meshLoc, RegressionDataElliptic & regressionData) : HBase<H<RegressionDataElliptic, Integrator, ORDER>, RegressionDataElliptic, Integrator, ORDER>(mesh, meshLoc, regressionData) {}
        RegressionDataElliptic createRegressionData(const TVector & anisoParam) const;
};

template <typename Integrator, UInt ORDER>
RegressionDataElliptic H<RegressionDataElliptic, Integrator, ORDER>::createRegressionData(const TVector & anisoParam) const {
    // Copy members of regressionData to pass them to its constructor
    std::vector<Point> locations  = H::regressionData_.getLocations();
    VectorXr observations = H::regressionData_.getObservations();
    Eigen::Matrix<Real, 2, 2> kappa = H::buildKappa(anisoParam);
    Eigen::Matrix<Real, 2, 1> beta = H::regressionData_.getBeta();
    MatrixXr covariates = H::regressionData_.getCovariates();
    std::vector<UInt> dirichletIndices = H::regressionData_.getDirichletIndices();
    std::vector<Real> dirichletValues = H::regressionData_.getDirichletValues();

    // Construct a new regressionData object with the desired kappa
    const RegressionDataElliptic data(
            locations,
            observations,
            H::regressionData_.getOrder(),
            H::regressionData_.getLambda(),
            kappa,
            beta,
            H::regressionData_.getC(),
            covariates,
            dirichletIndices,
            dirichletValues,
            H::regressionData_.computeDOF(),
            H::regressionData_.getGCVmethod(),
            H::regressionData_.getNrealizations());
    return data;
}
#endif
