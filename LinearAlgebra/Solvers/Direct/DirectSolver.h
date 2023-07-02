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

        shared_ptr<LinearSystem> getLinearSystem();
        
        virtual shared_ptr<MatrixDecomposition> getDecomposition();
        
    protected:
        
        shared_ptr<LinearSystem> _linearSystem;
        
        bool _storeDecompositionOnMatrix;
    };

} // LinearAlgebra

#endif //UNTITLED_DIRECTSOLVER_H
