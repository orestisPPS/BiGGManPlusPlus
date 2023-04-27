//
// Created by hal9000 on 4/24/23.
//

#include "LU.h"

namespace LinearAlgebra {

    LU::LU(bool storeDecompositionOnMatrix) : DirectSolver(storeDecompositionOnMatrix) {
        _l = nullptr;
        _u = nullptr;
    }
    
    LU::~LU() {
        if (_storeDecompositionOnMatrix) {
            delete _l;
            delete _u;
        }
    }

    void LU::solve() {
        if (!_storeDecompositionOnMatrix) {
            //Decompose the matrix in L and U
            auto lu = _linearSystem->matrix->LUdecomposition();
            auto l = get<0>(lu);
            auto u = get<1>(lu);
            //Solve the system
            //Forward substitution for Ly = RHS
            auto y = _forwardSubstitution(l, _linearSystem->RHS);
            //Backward substitution for Ux = y
            _linearSystem->solution = _backwardSubstitution(u, y);
        }
        else {
            //Decompose the matrix in L and U and store it in the original matrix
            _linearSystem->matrix->LUdecompositionOnMatrix();
            //Solve the system
            //Forward substitution for Ly = RHS
            auto y = _forwardSubstitution(_linearSystem->matrix, _linearSystem->RHS);
            //Backward substitution for Ux = y
            _linearSystem->solution = _backwardSubstitution(_linearSystem->matrix, y);
        }
    }
} // LinearAlgebra