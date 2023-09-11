//
// Created by hal9000 on 6/13/23.
//

#include "JacobiSolver.h"

namespace LinearAlgebra {

    JacobiSolver::JacobiSolver(double tolerance, unsigned maxIterations, VectorNormType normType,
                               unsigned userDefinedThreads, bool printOutput, bool throwExceptionOnMaxFailure) :
            StationaryIterative(tolerance, maxIterations, normType, userDefinedThreads, printOutput, throwExceptionOnMaxFailure) {
        _solverName = "Jacobi";
        _residualNorms = make_shared<list<double>>();
    }
    
    void JacobiSolver::_performMethodIteration(){
        auto n = _linearSystem->matrix->numberOfRows();
        for (unsigned row = 0; row < n; ++row) {
            double sum = 0.0;
            // sum = sum + A_ij * xOld_j
            //Sum the elements of the row except the diagonal element
            sum = _linearSystem->matrix->multiplyVectorRowWisePartial(_xOld, row, 0, row - 1);
            sum += _linearSystem->matrix->multiplyVectorRowWisePartial(_xOld, row, row + 1, n);

            // xNew_i = (b_i - sum) / A_ii
            _xNew->at(row) = (_linearSystem->rhs->at(row) - sum) / _linearSystem->matrix->getElement(row, row);
            // Calculate the _difference
            _difference->at(row) = _xNew->at(row) - _xOld->at(row);
            //Replace the old solution with the new one
            _xOld->at(row) = _xNew->at(row);
        }
    }

/*    void JacobiSolver::_threadJobJacobi(unsigned start, unsigned end) {
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
    }*/
    
    


} // LinearAlgebra