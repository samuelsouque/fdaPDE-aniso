#ifndef __ANISOTROPIC_SMOOTHING_IMP_H__
#define __ANISOTROPIC_SMOOTHING_IMP_H__

#include <utility>
#include <algorithm>
#include <limits>
#include "R_ext/Applic.h"
#include "j.h"

template <typename InputHandler, typename Integrator, UInt ORDER>
std::pair<const std::vector<VectorXr>, const typename H<InputHandler, Integrator, ORDER>::TVector> AnisotropicSmoothing<InputHandler, Integrator, ORDER>::smooth() const {
    using H = H<InputHandler, Integrator, ORDER>;

    const std::vector<Real>::size_type n_lambda = lambda_.size();
    std::vector<TVector, Eigen::aligned_allocator<TVector>> anisoParamSmooth(n_lambda);
    std::vector<Eigen::Index> crossValSmoothInd(n_lambda);
    std::vector<Real> gcvSmooth(n_lambda);

    constexpr int n = 2;
    double xin[n] = { M_PI_2, 5. }; // Starting parameter on entry
    double x[n]; // Final parameter on exit
    void *ex; // Extra arguments for the function, here we pass an H instance
    double val; // Final value 
    int fail; // End status (converged, ...)
    constexpr double abstol = -std::numeric_limits<double>::infinity();
    constexpr double reltol = std::sqrt(std::numeric_limits<double>::epsilon());
    constexpr double alpha = 1.; // Reflection factor for Nelder-Mead
    constexpr double beta = 0.5; // Contraction factor
    constexpr double gamma = 2.; // Expansion factor
    constexpr int trace = 0; // Type of diagnostic displayed
    int fncount; // Number of function calls
    constexpr int maxit = 500; 
    
    for(std::vector<Real>::size_type i = 0U; i < n_lambda; i++) {
        // Optimization of H
        regressionData_.setLambda(std::vector<Real>(1U, lambda_[i]));
        H h(mesh_, meshLoc_, regressionData_);

        if (i != 0U) {
            // Copy anisoParamSmooth[i-1] into xin
            Eigen::Map<TVector, EIGEN_MAX_ALIGN_BYTES>{xin} = anisoParamSmooth[i-1];
        }
        ex = &h;

        nmmin(n, xin, x, &val, H::fn, &fail, abstol, reltol, ex, alpha, beta, gamma, trace, &fncount, maxit);
        // Dealing with angle periodicity
        if (x[0] == H::upper(0)) {
            xin[0] = H::lower(0);
            nmmin(n, xin, x, &val, H::fn, &fail, abstol, reltol, ex, alpha, beta, gamma, trace, &fncount, maxit);
        }
        if (x[0] == H::lower(0)) {
            xin[0] = H::upper(0);
            nmmin(n, xin, x, &val, H::fn, &fail, abstol, reltol, ex, alpha, beta, gamma, trace, &fncount, maxit);
        }

        // Copy x into anisoParamSmooth[i]
        anisoParamSmooth[i] = Eigen::Map<TVector, EIGEN_MAX_ALIGN_BYTES>(x);
        
        if (fail) {
            REprintf("Nelder-Mead did not converged before %d iterations!\n", maxit);
        }

        // Computation of the GCV for current anisoParamSmooth[i]
        regressionData_.setLambda(lambdaCrossVal_);
        regressionData_.setComputeDOF(true);
        J<InputHandler, Integrator, ORDER> j(mesh_, meshLoc_, regressionData_);
        regressionData_.setComputeDOF(dof_);

        VectorXr gcvSeq = j.getGCV();

        Eigen::Index lambdaCrossValIndex;
        Real gcv = gcvSeq.minCoeff(&lambdaCrossValIndex);
        
        crossValSmoothInd[i] = lambdaCrossValIndex;
        gcvSmooth[i] = gcv;
    }
    // Choosing the best anisotropy matrix and lambda coefficient
    std::vector<Real>::const_iterator minIterator = std::min_element(gcvSmooth.cbegin(), gcvSmooth.cend());
    std::vector<Real>::difference_type optIndex = std::distance(gcvSmooth.cbegin(), minIterator);

    regressionData_.setLambda(std::vector<Real>(1U, lambdaCrossVal_[crossValSmoothInd[optIndex]]));
    regressionData_.setK(H::buildKappa(anisoParamSmooth[optIndex]));
    MixedFERegression<InputHandler, Integrator, ORDER, 2, 2> regressionFinal(mesh_, regressionData_);
    
    regressionFinal.apply();

    return std::make_pair(regressionFinal.getSolution(), anisoParamSmooth[optIndex]);
}

template <typename InputHandler, typename Integrator, UInt ORDER>
std::vector<Real> AnisotropicSmoothing<InputHandler, Integrator, ORDER>::seq(const UInt &n_obs, const Real &area) const {
    std::vector<Real> result{1.000000e-07, 1.258925e-07, 1.584893e-07, 1.995262e-07, 2.511886e-07, 3.162278e-07, 3.981072e-07, 5.011872e-07, 6.309573e-07, 7.943282e-07, 1.000000e-06, 1.258925e-06, 1.584893e-06, 1.995262e-06, 2.511886e-06, 3.162278e-06, 3.981072e-06, 5.011872e-06, 6.309573e-06, 7.943282e-06, 1.000000e-05, 1.258925e-05, 1.584893e-05, 1.995262e-05, 2.511886e-05, 3.162278e-05, 3.981072e-05, 5.011872e-05, 6.309573e-05, 7.943282e-05, 1.000000e-04, 1.258925e-04, 1.584893e-04, 1.995262e-04, 2.511886e-04, 3.162278e-04, 3.981072e-04, 5.011872e-04, 6.309573e-04, 7.943282e-04, 1.000000e-03, 1.258925e-03, 1.584893e-03, 1.995262e-03, 2.511886e-03, 3.162278e-03, 3.981072e-03, 5.011872e-03, 6.309573e-03, 7.943282e-03, 1.000000e-02, 1.258925e-02, 1.584893e-02, 1.995262e-02, 2.511886e-02, 3.162278e-02, 3.981072e-02, 5.011872e-02, 6.309573e-02, 7.943282e-02, 1.000000e-01, 1.258925e-01, 1.584893e-01, 1.995262e-01, 2.511886e-01, 3.162278e-01, 3.981072e-01, 5.011872e-01, 6.309573e-01, 7.943282e-01};
    std::transform(result.begin(), result.end(), result.begin(), [&n_obs, &area] (const Real &el) -> Real { return el/(1-el)*(n_obs/area); });
    return result;
}

#endif
