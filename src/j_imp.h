#ifndef __J_IMP_H__
#define __J_IMP_H__

#include "evaluatorExt.h"
#include "mesh_objects.h"

template <typename InputHandler, typename Integrator, UInt ORDER>
J<InputHandler, Integrator, ORDER>::J(const MeshHandler<ORDER, 2, 2> & mesh, const InputHandler & regressionData) : mesh_(mesh), regressionData_(regressionData), regression_(mesh_, regressionData_) {
    regression_.apply();
}

template <typename InputHandler, typename Integrator, UInt ORDER>
const std::vector<VectorXr> & J<InputHandler, Integrator, ORDER>::getSolution() const {
    return regression_.getSolution();
}

template <typename InputHandler, typename Integrator, UInt ORDER>
const std::vector<Real> & J<InputHandler, Integrator, ORDER>::getDOF() const {
    return regression_.getDOF();
}

template <typename InputHandler, typename Integrator, UInt ORDER>
VectorXr J<InputHandler, Integrator, ORDER>::getGCV() const {
    const std::vector<VectorXr> & solution = getSolution();

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

    EvaluatorExt<ORDER> evaluator(mesh_);
    const MatrixXr & fnhat = evaluator.eval(solution, locations);
    
    Real np = locations.size();
    const std::vector<Real> & edf = getDOF();

    for (std::size_t i=0; i<edf.size(); i++) {
        if (np - edf[i] <= 0) {
            REprintf("Some values of 'edf' are inconstistent. This might be due to ill-conditioning of the linear system. Try increasing value of 'lambda'.");
            break;
        }
    }
    
    using ArrayXXr = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic>;
    using ArrayXr = Eigen::Array<Real, Eigen::Dynamic, 1>;

    const VectorXr & observations = regressionData_.getObservations();

    const MatrixXr diff = fnhat.colwise() - observations;
    const ArrayXXr quotient = np / (np - Eigen::Map<const ArrayXr>(edf.data(), edf.size())).pow(2);

    VectorXr gcv = (quotient * (diff.transpose() * diff).diagonal().array()).matrix();

    return gcv;
}
#endif
