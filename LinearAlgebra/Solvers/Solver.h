//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_SOLVER_H
#define UNTITLED_SOLVER_H

#include "../Array/Array.h"
#include "../Norms/VectorNorm.h"
#include "../AnalysisLinearSystem.h"

namespace LinearAlgebra {
    
    enum SolverType{
        Direct,
        Iterative,
        PreconditionedIterative
    };

    class Solver {
        
    public:
        
        void setLinearSystem(AnalysisLinearSystem* linearSystem);

        SolverType type();

        virtual void solve();


    protected:
        AnalysisLinearSystem* _linearSystem;        
        
        SolverType _solverType;
    };

} // LinearAlgebra

#endif //UNTITLED_SOLVER_H
