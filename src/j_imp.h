#ifndef __J_IMP_H__
#define __J_IMP_H__

#include "evaluatorExt.h"
#include "mesh_objects.h"

template <typename InputHandler, typename Integrator, UInt ORDER>
J<InputHandler, Integrator, ORDER>::J(const MeshHandler<ORDER, 2, 2> &mesh, const std::vector<Point> &meshLoc, const InputHandler &regressionData) : mesh_(mesh), meshLoc_(meshLoc), regressionData_(regressionData), regression_(mesh_, regressionData_) {
    regression_.apply();
}

template <typename InputHandler, typename Integrator, UInt ORDER>
VectorXr J<InputHandler, Integrator, ORDER>::getGCV() const {
    const std::vector<VectorXr> &solution = regression_.getSolution();

    const std::vector<Point> &locations = regressionData_.isLocationsByNodes() ? meshLoc_ : regressionData_.getLocations();
    
    EvaluatorExt<ORDER> evaluator(mesh_);
    const MatrixXr &fnhat = evaluator.eval(solution, locations);
    
    Real np = locations.size();
    const std::vector<Real> &edf = regression_.getDOF();

    auto test_inconsistent = [&np] (const Real &edf) -> bool { return (np - edf <= 0); };
    std::vector<Real>::const_iterator any_inconsistent = std::find_if(edf.cbegin(), edf.cend(), test_inconsistent);
    if (any_inconsistent != edf.cend()) {
        REprintf("Some values of 'edf' are inconsistent. This might be due to ill-conditioning of the linear system. Try increasing value of 'lambda'.\n");
    }
    
    using ArrayXXr = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using ArrayXr = Eigen::Array<Real, Eigen::Dynamic, 1>;

    const VectorXr &observations = regressionData_.getObservations();

    const MatrixXr diff = fnhat.colwise() - observations;
    const ArrayXXr quotient = np / (np - Eigen::Map<const ArrayXr>(edf.data(), edf.size())).pow(2);

    VectorXr gcv = (quotient * (diff.transpose() * diff).diagonal().array()).matrix();

    return gcv;
}
#endif
