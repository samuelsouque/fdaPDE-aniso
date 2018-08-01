#ifndef __H_IMP_H__
#define __H_IMP_H__

#include "evaluatorExt.h"
#include "mixedFERegression.h"
#include "mesh_objects.h"

template <typename InputHandler, typename Integrator, UInt ORDER>
typename H<InputHandler, Integrator, ORDER>::TVector H<InputHandler, Integrator, ORDER>::lower(0., 1.);

template <typename InputHandler, typename Integrator, UInt ORDER>
typename H<InputHandler, Integrator, ORDER>::TVector H<InputHandler, Integrator, ORDER>::upper(M_PI, 1000.);

template <typename InputHandler, typename Integrator, UInt ORDER>
Real H<InputHandler, Integrator, ORDER>::value(const TVector &anisoParam) {
    /*Calculating fHat*/
    VectorXr estimations = fHat(anisoParam);

    /* Calculating H */
    estimations -= regressionData_.getObservations();
    Real H = estimations.squaredNorm();

    return H;
}

template <typename InputHandler, typename Integrator, UInt ORDER>
const VectorXr H<InputHandler, Integrator, ORDER>::fHat(const TVector &anisoParam) const {
    regressionData_.setK(buildKappa(anisoParam));
    // Compute the regression coefficients
    MixedFERegression<RegressionDataElliptic, Integrator, ORDER, 2, 2> regression(mesh_, regressionData_);
    regression.apply();
    const std::vector<VectorXr> &solution = regression.getSolution();
    
    // Return the evaluation of the coefficients at the data points
    EvaluatorExt<ORDER> evaluator(mesh_);
    const VectorXr result = evaluator.eval(solution, regressionData_.isLocationsByNodes() ? meshLoc_ : regressionData_.getLocations());

    return result;
}

template <typename InputHandler, typename Integrator, UInt ORDER>
Eigen::Matrix<Real, 2, 2> H<InputHandler, Integrator, ORDER>::buildKappa(const TVector &anisoParam) {
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

template <typename InputHandler, typename Integrator, UInt ORDER>
void H<InputHandler, Integrator, ORDER>::gradient(const TVector &anisoParam, TVector &grad) {
    constexpr Real eps = 1e-6;
    Real epsUsed1, epsUsed2;
    Real val1, val2;
    TVector tmp;

    for (Eigen::Index i=0U; i<anisoParam.size(); i++) {
        tmp = anisoParam;

        tmp(i) += eps;
        if (tmp(i) > upper(i)) {
            tmp(i) = upper(i);
            epsUsed1 = tmp(i) - anisoParam(i);
        } else {
            epsUsed1 = eps;
        }
        val1 = value(tmp);

        tmp(i) = anisoParam(i) - eps;
        if (tmp(i) < lower(i)) {
            tmp(i) = lower(i);
            epsUsed2 = anisoParam(i) - tmp(i);
        } else {
            epsUsed2 = eps;
        }
        val2 = value(tmp);

        grad(i) = (val1 - val2) / (epsUsed1 + epsUsed2);
    }
}

#endif
