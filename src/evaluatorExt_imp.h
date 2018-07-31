#ifndef __EVALUATOR_EXT_IMP_H__
#define __EVALUATOR_EXT_IMP_H__

template <UInt ORDER>
const MatrixXr EvaluatorExt<ORDER>::eval(const std::vector<VectorXr> &solution, const std::vector<Point> &locations) {
    // Sizes
    const std::vector<Point>::size_type numLoc = locations.size();
    const std::vector<VectorXr>::size_type numSol = solution.size();

    // Pointers
    Real *X = new Real[numLoc];
    Real *Y = new Real[numLoc];
    Real *coef = new Real[numNodes_];
    Real *res = new Real[numLoc];
    std::vector<bool> isInside(numLoc);

    // Bind the vector of 2D points into x and y's coordinates arrays
    for (std::vector<Point>::size_type i=0U; i<numLoc; i++) {
        Point p = locations[i];
        X[i] = p[0];
        Y[i] = p[1];
    }

    // Allocate a matrix for the result
    MatrixXr result(numLoc, numSol);

    for (std::vector<VectorXr>::size_type i=0U; i<numSol; i++) {
        // Map the values of solution[i] into the coef raw buffer
        Eigen::Map<VectorXr>(coef, numNodes_) = solution[i].head(numNodes_);

        // Perform the evaluation and assign the result in res
        Evaluator<ORDER, 2, 2>::eval(X, Y, numLoc, coef, 1, true, res, isInside);

        result.col(i) = Eigen::Map<VectorXr>(res, numLoc);
    }

    // ? Done in FEMeval.cpp
    for (std::vector<Point>::size_type i=0U; i<numLoc; i++) {
        if(!isInside[i]) {
            res[i] = std::numeric_limits<Real>::quiet_NaN();
        }
    }

    delete X;
    delete Y;
    delete res;
    delete coef;

    return result;
}

#endif
