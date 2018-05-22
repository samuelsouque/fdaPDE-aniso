#ifndef __EVALUATOR_EXT_H__
#define __EVALUATOR_EXT_H__

#include "fdaPDE.h"
#include "mesh.h"
#include "evaluator.h"

/**
 * @class EvaluatorExt
 * @brief allows to conveniently transform the solution vector of coefficients to the estimation fHat.
 */
template <UInt ORDER>
class EvaluatorExt : public Evaluator<ORDER, 2, 2> {

    public:
        EvaluatorExt(const MeshHandler<ORDER, 2, 2> & mesh) : Evaluator<ORDER, 2, 2>(mesh), numNodes_(mesh.num_nodes()) {}

        /**
		 * @param solution the vector of solution coefficients as returned by the method MixedFERegression::apply()
		 * @return The vector of estimations fHat.
		 */
        const MatrixXr eval(const std::vector<VectorXr> & solution, const std::vector<Point> & locations);

    private:
        const UInt numNodes_;
};

#include "evaluatorExt_imp.h"

#endif
