#ifndef __ANISOTROPIC_SMOOTHING_IMP_H__
#define __ANISOTROPIC_SMOOTHING_IMP_H__

#include <utility>
#include <algorithm>
#include "R_ext/Applic.h"
#include "j.h"

template <typename InputHandler, typename Integrator, UInt ORDER>
std::pair<const std::vector<VectorXr>, const typename H<InputHandler, Integrator, ORDER>::TVector> AnisotropicSmoothing<InputHandler, Integrator, ORDER>::smooth() const {
    const std::vector<Real>::size_type n_lambda = lambda_.size();
    std::vector<TVector, Eigen::aligned_allocator<TVector>> anisoParamSmooth(n_lambda, TVector(M_PI_2, 5.));
    std::vector<Eigen::Index> crossValSmoothInd(n_lambda);
    std::vector<Real> gcvSmooth(n_lambda);

    //#pragma omp parallel for
    for(std::vector<Real>::size_type i = 0U; i < n_lambda; i++) {
        // Optimization of H
        regressionData_.setLambda(std::vector<Real>(1U, lambda_[i]));
        H<InputHandler, Integrator, ORDER> h(mesh_, meshLoc_, regressionData_);

        if (i != 0U) {
            anisoParamSmooth[i] = anisoParamSmooth[i-1];
        }
        
        int lmm = 5; // Number of BFGS updates retained 
        double *lower = H<InputHandler, Integrator, ORDER>::lower.data();
        double *upper = H<InputHandler, Integrator, ORDER>::upper.data();
        int nbd[2]; nbd[0] = 2; nbd[1] = 2;// nbd[i]=2 for checking lower and upper bound on i-th variable
        double val; // Function value
        int fail; // End status (converged, ...)
        void *ex = &h; // Extra arguments for the function, here we pass an H instance
        double factr = 1e7; // Convergence criterion on the objective function
        double pgtol = 0.; // Convergence criterion on the function gradient
        int fncount; // Number of function calls
        int grcount; // Number of gradient calls
        int maxit = 10000; 
        char msg[60]; // Additional info
        int trace = 0; // Type of diagnostic displayed
        int nREPORT = 10; // Frequency of diagnostic when displayed

        lbfgsb(2, lmm, anisoParamSmooth[i].data(), lower, upper, nbd, &val, H<InputHandler, Integrator, ORDER>::fn, H<InputHandler, Integrator, ORDER>::gr, &fail, ex, factr, pgtol, &fncount, &grcount, maxit, msg, trace, nREPORT);

        // Dealing with angle periodicity
        if (anisoParamSmooth[i](0) == M_PI) {
            anisoParamSmooth[i](0) = 0.;
            lbfgsb(2, lmm, anisoParamSmooth[i].data(), lower, upper, nbd, &val, H<InputHandler, Integrator, ORDER>::fn, H<InputHandler, Integrator, ORDER>::gr, &fail, ex, factr, pgtol, &fncount, &grcount, maxit, msg, trace, nREPORT);
        }
        if (anisoParamSmooth[i](0) == 0.) {
            anisoParamSmooth[i](0) = M_PI;
            lbfgsb(2, lmm, anisoParamSmooth[i].data(), lower, upper, nbd, &val, H<InputHandler, Integrator, ORDER>::fn, H<InputHandler, Integrator, ORDER>::gr, &fail, ex, factr, pgtol, &fncount, &grcount, maxit, msg, trace, nREPORT);
        }
        
        //if (fail) {
        //    REprintf("L-BFGS-B did not converged: %s\n", msg);
        //}

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
    regressionData_.setK(H<InputHandler, Integrator, ORDER>::buildKappa(anisoParamSmooth[optIndex]));
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
