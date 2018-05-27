#ifndef __ANISOTROPIC_SMOOTHING_IMP_H__
#define __ANISOTROPIC_SMOOTHING_IMP_H__

#include <utility>
#include <algorithm>
#include "solver/lbfgsbsolver.h"
#include "j.h"

template <class Derived, typename InputHandler, typename Integrator, UInt ORDER>
std::pair<const std::vector<VectorXr>, const typename H<InputHandler, Integrator, ORDER>::TVector> AnisotropicSmoothingBase<Derived, InputHandler, Integrator, ORDER>::smooth() const {
    constexpr std::array<Real, 70>::size_type n_crossVal = lambdaCrossVal_.size();
    std::vector<TVector, Eigen::aligned_allocator<TVector>> kSmooth(n_crossVal, TVector(M_PI_2, 5.));
    std::vector<Eigen::Index> rhoSmoothInd(n_crossVal);
    std::vector<Real> gcvSmooth(n_crossVal);

    #pragma omp parallel for
    for(std::array<Real, 70>::size_type i = 0U; i < n_crossVal; i++) {
        // Optimization of H
        InputHandler dataUniqueLambda = createRegressionData(lambdaCrossVal_[i]);
        H<InputHandler, Integrator, ORDER> h(mesh_, dataUniqueLambda);

        cppoptlib::LbfgsbSolver<H<InputHandler, Integrator, ORDER>> solver;
        solver.minimize(h, kSmooth[i]);

        // Dealing with angle periodicity
        if (kSmooth[i](0) == M_PI) {
            kSmooth[i](0) = 0.;
            solver.minimize(h, kSmooth[i]);
            REprintf("Angle equal to pi at iteration = %3u\n", i);
        }
        if (kSmooth[i](0) == 0.) {
            kSmooth[i](0) = M_PI;
            solver.minimize(h, kSmooth[i]);
            REprintf("Angle equal to 0 at iteration = %3u\n", i);
        }

        // Computation of the GCV for current k
        InputHandler dataSelectedK = createRegressionData(kSmooth[i], true);
        J<InputHandler, Integrator, ORDER> j(mesh_, dataSelectedK);
        
        VectorXr gcvSeq = j.getGCV();
        Eigen::Index rhoIndex;
        Real gcv = gcvSeq.minCoeff(&rhoIndex);

        //kSmooth[i] = k;
        rhoSmoothInd[i] = rhoIndex;
        gcvSmooth[i] = gcv;

    }

    // Choosing the best anisotropy matrix and lambda coefficient
    std::vector<Real>::const_iterator minIterator = std::min_element(gcvSmooth.cbegin(), gcvSmooth.cend());
    std::vector<Real>::difference_type optIndex = std::distance(gcvSmooth.cbegin(), minIterator);

    InputHandler regressionDataFinal = createRegressionData(regressionData_.getLambda()[rhoSmoothInd[optIndex]], kSmooth[optIndex]);
    MixedFERegression<InputHandler, Integrator, ORDER, 2, 2> regressionFinal(mesh_, regressionDataFinal);
    
    regressionFinal.apply();

    return std::make_pair(regressionFinal.getSolution(), kSmooth[optIndex]);
}

template <typename Integrator, UInt ORDER>
struct AnisotropicSmoothing<RegressionDataElliptic, Integrator, ORDER> : public AnisotropicSmoothingBase<AnisotropicSmoothing<RegressionDataElliptic, Integrator, ORDER>, RegressionDataElliptic, Integrator, ORDER> {
        using typename AnisotropicSmoothingBase<AnisotropicSmoothing, RegressionDataElliptic, Integrator, ORDER>::TVector;
        AnisotropicSmoothing(const RegressionDataElliptic & regressionData, const MeshHandler<ORDER, 2, 2> & mesh) : AnisotropicSmoothingBase<AnisotropicSmoothing, RegressionDataElliptic, Integrator, ORDER>(regressionData, mesh) {}

        RegressionDataElliptic createRegressionData(const Real & lambda) const;
        RegressionDataElliptic createRegressionData(const TVector & k, const bool dof) const;
        RegressionDataElliptic createRegressionData(const Real & lambda, const TVector & k) const;
};

template <typename Integrator, UInt ORDER>
RegressionDataElliptic AnisotropicSmoothing<RegressionDataElliptic, Integrator, ORDER>::createRegressionData(const Real & lambda) const {
    std::vector<Point> locations = AnisotropicSmoothing::regressionData_.getLocations();
    VectorXr observations = AnisotropicSmoothing::regressionData_.getObservations();
    std::vector<Real> uniqueLambda(1U, lambda);
    Eigen::Matrix<Real, 2, 2> kappa;
    Eigen::Matrix<Real, 2, 1> beta = AnisotropicSmoothing::regressionData_.getBeta();
    MatrixXr covariates = AnisotropicSmoothing::regressionData_.getCovariates();
    std::vector<UInt> dirichletIndices = AnisotropicSmoothing::regressionData_.getDirichletIndices();
    std::vector<Real> dirichletValues = AnisotropicSmoothing::regressionData_.getDirichletValues();

    const RegressionDataElliptic result(
            locations,
            observations,
            AnisotropicSmoothing::regressionData_.getOrder(),
            uniqueLambda,
            kappa,
            beta,
            AnisotropicSmoothing::regressionData_.getC(),
            covariates,
            dirichletIndices,
            dirichletValues,
            AnisotropicSmoothing::regressionData_.computeDOF());

    return result;
}

template <typename Integrator, UInt ORDER>
RegressionDataElliptic AnisotropicSmoothing<RegressionDataElliptic, Integrator, ORDER>::createRegressionData(const TVector & k, const bool dof) const {
    std::vector<Point> locations = AnisotropicSmoothing::regressionData_.getLocations();
    VectorXr observations = AnisotropicSmoothing::regressionData_.getObservations();
    Eigen::Matrix<Real, 2, 2> kappa = H<RegressionDataElliptic, Integrator, ORDER>::buildKappa(k);
    Eigen::Matrix<Real, 2, 1> beta = AnisotropicSmoothing::regressionData_.getBeta();
    MatrixXr covariates = AnisotropicSmoothing::regressionData_.getCovariates();
    std::vector<UInt> dirichletIndices = AnisotropicSmoothing::regressionData_.getDirichletIndices();
    std::vector<Real> dirichletValues = AnisotropicSmoothing::regressionData_.getDirichletValues();

    const RegressionDataElliptic result(
            locations,
            observations,
            AnisotropicSmoothing::regressionData_.getOrder(),
            AnisotropicSmoothing::regressionData_.getLambda(),
            kappa,
            beta,
            AnisotropicSmoothing::regressionData_.getC(),
            covariates,
            dirichletIndices,
            dirichletValues,
            dof);

    return result;
}

template <typename Integrator, UInt ORDER>
RegressionDataElliptic AnisotropicSmoothing<RegressionDataElliptic, Integrator, ORDER>::createRegressionData(const Real & lambda, const TVector & k) const {
    std::vector<Point> locations = AnisotropicSmoothing::regressionData_.getLocations();
    VectorXr observations = AnisotropicSmoothing::regressionData_.getObservations();
    std::vector<Real> uniqueLambda(1U, lambda);
    Eigen::Matrix<Real, 2, 2> kappa = H<RegressionDataElliptic, Integrator, ORDER>::buildKappa(k);
    Eigen::Matrix<Real, 2, 1> beta = AnisotropicSmoothing::regressionData_.getBeta();
    MatrixXr covariates = AnisotropicSmoothing::regressionData_.getCovariates();
    std::vector<UInt> dirichletIndices = AnisotropicSmoothing::regressionData_.getDirichletIndices();
    std::vector<Real> dirichletValues = AnisotropicSmoothing::regressionData_.getDirichletValues();

    const RegressionDataElliptic result(
            locations,
            observations,
            AnisotropicSmoothing::regressionData_.getOrder(),
            uniqueLambda,
            kappa,
            beta,
            AnisotropicSmoothing::regressionData_.getC(),
            covariates,
            dirichletIndices,
            dirichletValues,
            AnisotropicSmoothing::regressionData_.computeDOF());

    return result;
}

#endif
