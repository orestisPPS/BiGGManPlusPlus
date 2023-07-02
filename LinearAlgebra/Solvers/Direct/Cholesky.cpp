//
// Created by hal9000 on 4/25/23.
//

#include "Cholesky.h"

namespace LinearAlgebra {

    Cholesky::Cholesky(bool storeDecompositionOnMatrix) : DirectSolver(storeDecompositionOnMatrix) {
        _l = nullptr;
        _lT = nullptr;
    }
    
    void Cholesky::solve() {
/*        if (!_storeDecompositionOnMatrix) {
            //Decompose the matrix in L and L^T
            auto cholesky = _linerSystem->matrix->CholeskyDecomposition();
            _l = get<0>(cholesky);
            _lT = get<1>(cholesky);
            //Solve the system
            //Forward substitution for Ly = rhs
            auto y = _forwardSubstitution(_l, _linearSystem->rhs);
            //Backward substitution for L^Tx = y
            _linearSystem->solution = _backwardSubstitution(_lT, y);
        } else {
            //Decompose the matrix in L and L^T and store it in the original matrix
            _linearSystem->matrix->CholeskyDecompositionOnMatrix();
            //Solve the system
            //Forward substitution for Ly = rhs
            auto y = _forwardSubstitution(_linearSystem->matrix, _linearSystem->rhs);
            //Backward substitution for L^Tx = y
            _linearSystem->solution = _backwardSubstitution(_linearSystem->matrix, y);
        }*/
    }
} // LinearAlgebra

