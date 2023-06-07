//
// Created by hal9000 on 4/25/23.
//

#include "DirectSolver.h"

namespace LinearAlgebra {
    
    DirectSolver::DirectSolver(bool storeDecompositionOnMatrix) :
                    Solver(),
                    _storeDecompositionOnMatrix(storeDecompositionOnMatrix),
                    _linearSystem(nullptr){}

    unique_ptr<LinearSystem> DirectSolver::getLinearSystem() {
        if (_linearSystem == nullptr) {
            throw runtime_error("LinearSystem is not set");
        }
        return unique_ptr<LinearSystem>(_linearSystem);
    }
    
    unique_ptr<MatrixDecomposition> DirectSolver::getDecomposition() {
        return nullptr;
    }
    
    
} // LinearAlgebra