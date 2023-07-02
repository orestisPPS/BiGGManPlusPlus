//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_SOLVER_H
#define UNTITLED_SOLVER_H


#include "../Norms/VectorNorm.h"
#include "../LinearSystem.h"

namespace LinearAlgebra {
    
    enum SolverType{
        Direct,
        Iterative,
        PreconditionedIterative
    };

    class Solver {
        
    public:

        virtual void setLinearSystem(shared_ptr<LinearSystem> linearSystem);

        SolverType type();

        virtual void solve();


    protected:
        shared_ptr<LinearSystem> _linearSystem;        
        
        SolverType _solverType;

        bool _isLinearSystemSet;

    };

} // LinearAlgebra

#endif //UNTITLED_SOLVER_H
