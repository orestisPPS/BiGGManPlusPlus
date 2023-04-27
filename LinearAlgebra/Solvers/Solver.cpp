//
// Created by hal9000 on 4/18/23.
//

#include "Solver.h"

namespace LinearAlgebra {
    
    void Solver::setLinearSystem(AnalysisLinearSystem* linearSystem){
        _linearSystem = linearSystem;
    }
    
    SolverType Solver::type(){
        return _solverType;
    }
    
    void Solver::solve(){}

} // LinearAlgebra