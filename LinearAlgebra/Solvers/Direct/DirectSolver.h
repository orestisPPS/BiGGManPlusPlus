//
// Created by hal9000 on 4/25/23.
//

#ifndef UNTITLED_DIRECTSOLVER_H
#define UNTITLED_DIRECTSOLVER_H

#include "../Solver.h"
#include "../../Array/DecompositionMethods/MatrixDecomposition.h"

namespace LinearAlgebra {

    class DirectSolver : public Solver {
        
    public:
        
        explicit DirectSolver(bool storeDecompositionOnMatrix);

        unique_ptr<LinearSystem> getLinearSystem();

        virtual void setLinearSystem(LinearSystem* linearSystem);

        virtual unique_ptr<MatrixDecomposition> getDecomposition();
        
        virtual void solve();
        
    protected:
        
        LinearSystem* _linearSystem;
        
        bool _storeDecompositionOnMatrix;
    };

} // LinearAlgebra

#endif //UNTITLED_DIRECTSOLVER_H
