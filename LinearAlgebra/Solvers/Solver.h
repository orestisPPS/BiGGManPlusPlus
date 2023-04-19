//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_SOLVER_H
#define UNTITLED_SOLVER_H

#include "../Array.h"
#include "../Norms/VectorNorm.h"

namespace LinearAlgebra {
    
    enum SolverType{
        Direct,
        Iterative,
        PreconditionedIterative
    };

    class Solver {
        
        Solver(Array<double>* Matrix, vector<double>* RHS);
        
        SolverType type();
        
    private:
        Array<double>* _matrix;
        
        vector<double>* _rhs;
        
        SolverType _solverType;
        
    };

} // LinearAlgebra

#endif //UNTITLED_SOLVER_H
