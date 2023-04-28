//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_SOLVER_H
#define UNTITLED_SOLVER_H

#include "../Array/Array.h"
#include "../Norms/VectorNorm.h"
#include "../AnalysisLinearSystemInitializer.h"

namespace LinearAlgebra {
    
    enum SolverType{
        Direct,
        Iterative,
        PreconditionedIterative
    };

    class Solver {
        
    public:
        
        void setLinearSystem(AnalysisLinearSystemInitializer* linearSystem);

        SolverType type();

        virtual void solve();


    protected:
        AnalysisLinearSystemInitializer* _linearSystem;        
        
        SolverType _solverType;
    };

} // LinearAlgebra

#endif //UNTITLED_SOLVER_H
