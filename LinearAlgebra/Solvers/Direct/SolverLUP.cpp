//
// Created by hal9000 on 4/24/23.
//

#include "SolverLUP.h"

namespace LinearAlgebra {

    SolverLUP::SolverLUP(double  pivotTolerance, bool storeDecompositionOnMatrix) :
                DirectSolver(storeDecompositionOnMatrix),
                _decomposition(nullptr),
                _pivotTolerance(pivotTolerance),
                _throwExceptionOnSingularMatrix(true){}
    
    SolverLUP::SolverLUP(double pivotTolerance, bool storeDecompositionOnMatrix, bool throwExceptionOnSingularMatrix) :
                DirectSolver(storeDecompositionOnMatrix),
                _decomposition(nullptr),
                _pivotTolerance(pivotTolerance),
                _throwExceptionOnSingularMatrix(throwExceptionOnSingularMatrix){}
                
    SolverLUP::~SolverLUP() {
        delete _decomposition;
    }

    void SolverLUP::setLinearSystem(LinearSystem* linearSystem) {
        _linearSystem = linearSystem;
        _decomposition = new DecompositionLUP(_linearSystem->matrix, _pivotTolerance, _throwExceptionOnSingularMatrix);

    }

    unique_ptr<MatrixDecomposition> SolverLUP::getDecomposition() {
        return unique_ptr<MatrixDecomposition>(_decomposition);
    }
    
    void SolverLUP::solve() {
        if (_storeDecompositionOnMatrix) {
            cout<<"Decomposition Initiated..."<<endl;
            _decomposition->decomposeOnMatrix();
        }
        else
        {
            _decomposition->decompose(true);
        }
        _linearSystem->solution = _decomposition->solve(_linearSystem->RHS);
    }
} // LinearAlgebra
