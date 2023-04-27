//
// Created by hal9000 on 4/25/23.
//

#include "DirectSolver.h"

namespace LinearAlgebra {
    
    DirectSolver::DirectSolver(bool storeDecompositionOnMatrix) : Solver() {
        _storeDecompositionOnMatrix = storeDecompositionOnMatrix;
        _solverType = SolverType::Direct;

    }


    vector<double>* DirectSolver:: _forwardSubstitution(Array<double>* L, const vector<double>* RHS) {
     unsigned int n = L->numberOfRows();
     auto y = new vector<double>(n, 0.0);
     for (int i = 0; i < n; ++i) {
         y->at(i) = RHS->at(i);
         for (int j = 0; j < i; ++j) {
             y->at(i) -= L->at(i, j) * y->at(j);
         }
         y->at(i) /= L->at(i, i);
     }   
     return y;
    }
     
    vector<double>* DirectSolver:: _backwardSubstitution(Array<double>* U, const vector<double>* RHS) {
        unsigned int n = U->numberOfRows();
        auto y = new vector<double>(n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            y->at(i) = RHS->at(i);
            for (int j = i + 1; j < n; ++j) {
                y->at(i) -= U->at(i, j) * y->at(j);
            }
            y->at(i) /= U->at(i, i);
        }
        return y;
    }
} // LinearAlgebra