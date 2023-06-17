//
// Created by hal9000 on 6/13/23.
//

#include "JacobiSolver.h"

namespace LinearAlgebra {

    JacobiSolver::JacobiSolver(bool vTechKickInYoo, VectorNormType normType, double tolerance, unsigned maxIterations,
                               bool throwExceptionOnMaxFailure) :
            StationaryIterative(vTechKickInYoo, normType, tolerance, maxIterations, throwExceptionOnMaxFailure) {
        _solverName = "Jacobi";
        _residualNorms = make_shared<list<double>>();
    }



    void JacobiSolver::_threadJob(unsigned start, unsigned end) {
        unsigned n = _linearSystem->matrix->numberOfRows();
        for (unsigned row = start; row < end; ++row) {
            double sum = 0.0;
            for (unsigned j = 0; j < n; j++) {
                if (row != j) {
                    // sum = sum + A_ij * xOld_j
                    sum += _linearSystem->matrix->at(row, j) * _xOld->at(j);
                }
            }
            // xNew_i = (b_i - sum) / A_ii
            _xNew->at(row) = (_linearSystem->rhs->at(row) - sum) / _linearSystem->matrix->at(row, row);
            // Calculate the _difference
            _difference->at(row) = _xNew->at(row) - _xOld->at(row);
            //Replace the old solution with the new one
            _xOld->at(row) = _xNew->at(row);
        }
    }

} // LinearAlgebra