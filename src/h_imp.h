#ifndef __H_IMP_H__
#define __H_IMP_H__

#include "evaluatorExt.h"
#include "mixedFERegression.h"
#include "mesh_objects.h"

template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
Real HBase<Derived, InputHandler, Integrator, ORDER>::value(const TVector & x) {
    TVector k = x.cwiseMax(TVector(0., 1.)).cwiseMin(TVector(M_PI, 1000));

    /*Calculating fHat*/
    VectorXr estimations = fHat(k);

    /* Calculating H */
    estimations -= regressionData_.getObservations();
    Real H = estimations.squaredNorm();

    if (k!=x) {
        H += 1000 * (k - x).squaredNorm();
    }
    return H;
}

template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
const VectorXr HBase<Derived, InputHandler, Integrator, ORDER>::fHat(const TVector & x) const {
    const InputHandler & data = createRegressionData(x);
    // Compute the regression coefficients
    MixedFERegression<RegressionDataElliptic, Integrator, ORDER, 2, 2> regression(mesh_, data);
    regression.apply();
    const std::vector<VectorXr> & solution = regression.getSolution();

    std::vector<Point> locations;
    if (regressionData_.isLocationsByNodes()) {
        const std::vector<UInt> & observationsIndices = regressionData_.getObservationsIndices();
        locations.reserve(observationsIndices.size());
        for (UInt i = 0; i < observationsIndices.size(); i++) {
            Id id = observationsIndices[i];
            Point point = mesh_.getPoint(id);
            locations.push_back(point);
        }
    } else {
        locations = regressionData_.getLocations();
    }

    // Return the evaluation of the coefficients at the data points
    EvaluatorExt<ORDER> evaluator(mesh_);
    const VectorXr result = evaluator.eval(solution, locations);

    return result;
}

template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
Eigen::Matrix<Real, 2, 2> HBase<Derived, InputHandler, Integrator, ORDER>::buildKappa(const TVector & x) {
    // x[0] is the angle α (alpha)
    // x[1] is the intensity γ (gamma)

    //Building K from x as an Eigen matrix
    Eigen::Matrix<Real,2,2> Q;
    Q << std::cos(x[0]), -std::sin(x[0]),
      std::sin(x[0]), std::cos(x[0]);

    Eigen::Matrix<Real,2,2> Sigma;
    Sigma << 1/std::sqrt(x[1]), 0,
          0, x[1]/std::sqrt(x[1]);

    const Eigen::Matrix<Real, 2, 2> Kappa = Q * Sigma * Q.inverse();
    return Kappa;
}

template <typename Integrator, UInt ORDER>
struct H<RegressionDataElliptic, Integrator, ORDER> : public HBase<H<RegressionDataElliptic, Integrator, ORDER>, RegressionDataElliptic, Integrator, ORDER> {
        using typename HBase<H<RegressionDataElliptic, Integrator, ORDER>, RegressionDataElliptic, Integrator, ORDER>::TVector;
        H(const MeshHandler<ORDER, 2, 2> & mesh, const RegressionDataElliptic & regressionData) : HBase<H<RegressionDataElliptic, Integrator, ORDER>, RegressionDataElliptic, Integrator, ORDER>(mesh, regressionData) {}
        RegressionDataElliptic createRegressionData(const TVector & x) const;
};

template <typename Integrator, UInt ORDER>
RegressionDataElliptic H<RegressionDataElliptic, Integrator, ORDER>::createRegressionData(const TVector & x) const {
    // Copy members of regressionData to pass them to its constructor
    std::vector<Point> locations  = H::regressionData_.getLocations();
    VectorXr observations = H::regressionData_.getObservations();
    Eigen::Matrix<Real, 2, 2> kappa = H::buildKappa(x);
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
