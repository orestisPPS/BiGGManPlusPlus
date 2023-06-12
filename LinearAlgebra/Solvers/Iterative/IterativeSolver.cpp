//
// Created by hal9000 on 4/18/23.
//

#include "IterativeSolver.h"

#include <memory>

namespace LinearAlgebra {
    
/*
    auto x = VectorNorm::_calculateLInfNorm(new vector<double>());
*/

    IterativeSolver::IterativeSolver(VectorNormType normType, double tolerance, unsigned maxIterations) : 
        Solver(), _normType(normType), _tolerance(tolerance), _maxIterations(maxIterations), _isLinearSystemSet(false), _isInitialized(false) {
        _xInitial = nullptr;
        _xNew = nullptr;
        _xOld = nullptr;
    }
    
    IterativeSolver::~IterativeSolver(){
        _xInitial.reset();
        _xNew.reset();
        _xOld.reset();
    }
    
    void IterativeSolver::setTolerance(double tolerance){
        _tolerance = tolerance;
    }
    
    const double& IterativeSolver::getTolerance() const{
        return _tolerance;
    }
    
    
    void IterativeSolver::setMaxIterations(unsigned maxIterations){
        _maxIterations = maxIterations;
    }
    
    const unsigned& IterativeSolver::getMaxIterations() const{
        return _maxIterations;
    }
    
    void IterativeSolver::setNormType(VectorNormType normType){
        _normType = normType;
    }
    
    const VectorNormType& IterativeSolver::getNormType() const{
        return _normType;
    }
    
    void IterativeSolver::setLinearSystem(LinearSystem* linearSystem){
        _linearSystem = linearSystem;
        _isLinearSystemSet = true;
    }
    
    void IterativeSolver::setInitialSolution(unique_ptr<vector<double>> initialValue){
        if (!_isLinearSystemSet)
            throw std::invalid_argument("Linear system must be set before setting the initial solution.");
        if (initialValue->size() != _linearSystem->RHS->size()){
            throw std::invalid_argument("Initial solution vector must have the same size as the RHS vector.");
        }
        _xInitial = std::move(initialValue);
    }
    
    void IterativeSolver::setInitialSolution(double initialValue){
        if (!_isLinearSystemSet)
            throw std::invalid_argument("Linear system must be set before setting the initial solution.");
        _xInitial = make_unique<vector<double>>(_linearSystem->RHS->size(), initialValue);
    }
    
        
    
    
} // LinearAlgebra