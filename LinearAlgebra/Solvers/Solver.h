//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_SOLVER_H
#define UNTITLED_SOLVER_H

#include "../LinearSystem.h"

namespace LinearAlgebra {
    
    enum SolverType{
        Direct,
        Iterative,
        PreconditionedIterative
    };

    class Solver {
        
    public:
        
        void setLinearSystem(shared_ptr<LinearSystem> linearSystem);

        SolverType type();

        virtual void solve();

        virtual void setInitialSolution(shared_ptr<NumericalVector<double>> initialValue);

        virtual void setInitialSolution(double initialValue);
        
    protected:
        
        virtual void _initializeVectors();
        
        shared_ptr<LinearSystem> _linearSystem;

        bool _linearSystemInitialized;
        
        bool _vectorsInitialized;
        
        bool _solutionSet;
        
        SolverType _solverType;

        bool _isLinearSystemSet;

    };

} // LinearAlgebra

#endif //UNTITLED_SOLVER_H
