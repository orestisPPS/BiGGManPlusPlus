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
        
        void setLinearSystem(LinearSystem* linearSystem);

        SolverType type();

        virtual void solve();


    protected:
        LinearSystem* _linearSystem;        
        
        SolverType _solverType;
    };

} // LinearAlgebra

#endif //UNTITLED_SOLVER_H
