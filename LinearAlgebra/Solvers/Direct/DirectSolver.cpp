//
// Created by hal9000 on 4/25/23.
//

#include "DirectSolver.h"

namespace LinearAlgebra {
    
    DirectSolver::DirectSolver(bool storeDecompositionOnMatrix) :
                    Solver(),
                    _storeDecompositionOnMatrix(storeDecompositionOnMatrix),
                    _linearSystem(nullptr){}

    shared_ptr<LinearSystem> DirectSolver::getLinearSystem() {
        if (_linearSystem == nullptr) {
            throw runtime_error("LinearSystem is not set");
        }
        return _linearSystem;
    }
    
    shared_ptr<MatrixDecomposition> DirectSolver::getDecomposition() {
        return nullptr;
    }
    
    
} // LinearAlgebra