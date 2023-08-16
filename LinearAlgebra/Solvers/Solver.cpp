//
// Created by hal9000 on 4/18/23.
//

#include "Solver.h"

#include <utility>

namespace LinearAlgebra {
    
    SolverType Solver::type(){
        return _solverType;
    }
    
    void Solver::solve(){}
    
    void Solver::setLinearSystem(shared_ptr<LinearSystem> linearSystem) {
        _linearSystem = std::move(linearSystem);
        _isLinearSystemSet = true;
        _initializeVectors();
    }


    void Solver::setInitialSolution(shared_ptr<vector<double>> initialValue) {
    }

    void Solver::setInitialSolution(double initialValue) {

    }
    
    void Solver::_initializeVectors() { }

} // LinearAlgebra