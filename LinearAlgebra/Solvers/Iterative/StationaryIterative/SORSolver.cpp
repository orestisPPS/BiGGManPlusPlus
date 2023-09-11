//
// Created by hal9000 on 6/14/23.
//

#include "SORSolver.h"
namespace LinearAlgebra{
    SORSolver::SORSolver(double relaxationParameter, double tolerance, unsigned maxIterations, VectorNormType normType,
                         unsigned userDefinedThreads, bool printOutput, bool throwExceptionOnMaxFailure)
                : StationaryIterative(tolerance, maxIterations, normType, userDefinedThreads, printOutput, throwExceptionOnMaxFailure) {
        _relaxationParameter = relaxationParameter;
        _residualNorms = make_shared<list<double>>();
        _solverName = "SOR";
    }
    
    void SORSolver::_performMethodIteration() {
        auto n = _linearSystem->matrix->numberOfRows();
        for (unsigned row = 0; row < n; ++row) {
            double sum = 0.0;
            // sum = sum + A_ij * xOld_j
            //Sum the elements of the row except the diagonal element
            sum = _linearSystem->matrix->multiplyVectorRowWisePartial(_xNew, row, 0, row - 1);
            sum += _linearSystem->matrix->multiplyVectorRowWisePartial(_xOld, row, row + 1, n);

            // xNew_i = (b_i - w * sum) / A_ii
            auto gsPart = (_linearSystem->rhs->at(row) - sum) * (_relaxationParameter  / _linearSystem->matrix->getElement(row, row));
            _xNew->at(row) = (1.0 - _relaxationParameter) * _xOld->at(row) + gsPart;
            // Calculate the _difference
            _difference->at(row) = _xNew->at(row) - _xOld->at(row);
            //Replace the old solution with the new one
            _xOld->at(row) = _xNew->at(row);
        }
    }
/*    
    void SORSolver::_threadJobSOR(unsigned start, unsigned end) {
        unsigned n = _linearSystem->matrix->numberOfRows();
        for (unsigned row = start; row < end; ++row) {
            double sum = 0.0;
            for (unsigned j = 0; j < row; j++) {
                // sum = sum + A_ij * xNew_j
                sum += _linearSystem->matrix->at(row, j) * _xNew->at(j);
            }
            for (unsigned j = row + 1; j < n; j++) {
                // sum = sum + A_ij * xOld_j
                sum += _linearSystem->matrix->at(row, j) * _xOld->at(j);
            }

            // xNew_i = (b_i - w * sum) / A_ii
            auto gsPart = (_linearSystem->rhs->at(row) - sum) * (_relaxationParameter  / _linearSystem->matrix->at(row, row));
            _xNew->at(row) = (1.0 - _relaxationParameter) * _xOld->at(row) + gsPart;
            // Calculate the _difference
            _difference->at(row) = _xNew->at(row) - _xOld->at(row);
            //Replace the old solution with the new one
            _xOld->at(row) = _xNew->at(row);
        }
    }*/
}

// LinearAlgebra  