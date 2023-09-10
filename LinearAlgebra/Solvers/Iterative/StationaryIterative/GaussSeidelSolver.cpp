//
// Created by hal9000 on 6/13/23.
//

#include "GaussSeidelSolver.h"

namespace LinearAlgebra {

    GaussSeidelSolver::GaussSeidelSolver(double tolerance, unsigned maxIterations, VectorNormType normType,
                                         unsigned userDefinedThreads, bool printOutput, bool throwExceptionOnMaxFailure)
            : StationaryIterative(tolerance, maxIterations, normType, userDefinedThreads, printOutput,
                                  throwExceptionOnMaxFailure) {
        _residualNorms = make_shared<list<double>>();
        _solverName = "Gauss-Seidel";
    }

    void GaussSeidelSolver::_performMethodIteration() {
        _threadJobGaussSeidel(0, _linearSystem->matrix->numberOfRows());
    }

    void GaussSeidelSolver::_multiThreadSolution(const unsigned short &availableThreads,
                                         const unsigned short &numberOfRows) {

        //Initiate the thread pool map
        map<unsigned, thread> threadPool = map<unsigned, thread>();
        for (unsigned int i = 0; i < availableThreads; ++i) {
            threadPool.insert(pair<unsigned, thread>(i, thread()));
        }

        // Calculate the number of rows to assign to each thread
        unsigned rowsPerThread = numberOfRows / availableThreads;
        unsigned lastRows = numberOfRows % availableThreads;

        // Launch the threads and assign the work
        unsigned startRow = 0;
        unsigned endRow = 0;
        for (unsigned i = 0; i < availableThreads; ++i) {
            endRow = startRow + rowsPerThread;
            if (i == availableThreads - 1) {
                endRow = endRow + lastRows;
            }
            threadPool[i] = thread(&GaussSeidelSolver::_threadJobGaussSeidel, this, startRow, endRow);
            startRow = endRow;
        }
        // Wait for all the threads to finish
        for (auto &thread: threadPool) {
            thread.second.join();
        }
    }
    
    void GaussSeidelSolver::_threadJobGaussSeidel(unsigned int start, unsigned int end) {
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
            // xNew_i = (b_i - sum) / A_ii
            _xNew->at(row) = (_linearSystem->rhs->at(row) - sum) / _linearSystem->matrix->at(row, row);
            // Calculate the _difference
            _difference->at(row) = _xNew->at(row) - _xOld->at(row);
            //Replace the old solution with the new one
            _xOld->at(row) = _xNew->at(row);
        }
    }
    

} // LinearAlgebra