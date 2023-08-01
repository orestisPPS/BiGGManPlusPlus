//
// Created by hal9000 on 4/24/23.
//

#ifndef UNTITLED_DECOMPOSITIONLUP_H
#define UNTITLED_LUP_H

#include "DirectSolver.h"
#include "../../Array/DecompositionMethods/DecompositionLUP.h"

namespace LinearAlgebra {

    class SolverLUP : public DirectSolver {
        
    public:
        
        SolverLUP(double pivotTolerance, bool storeDecompositionOnMatrix);
        
        SolverLUP(double pivotTolerance, bool storeDecompositionOnMatrix, bool throwExceptionOnSingularMatrix);
        
        ~SolverLUP();
        
        shared_ptr<MatrixDecomposition> getDecomposition() override;
        
        void solve() override;
        
    private:
        
        DecompositionLUP* _decomposition;
        
        double _pivotTolerance;
        
        bool _throwExceptionOnSingularMatrix;

    };

} // LinearAlgebra

#endif //UNTITLED_DECOMPOSITIONLUP_H
