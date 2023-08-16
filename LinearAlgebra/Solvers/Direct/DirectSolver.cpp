//
// Created by hal9000 on 4/25/23.
//

#include "DirectSolver.h"

namespace LinearAlgebra {
    
    DirectSolver::DirectSolver(bool storeDecompositionOnMatrix) :
                    Solver(),
                    _storeDecompositionOnMatrix(storeDecompositionOnMatrix),
                    _linearSystem(nullptr){
        _linearSystemInitialized = false;
        _vectorsInitialized = false;
    }

    shared_ptr<LinearSystem> DirectSolver::getLinearSystem() {
        if (_linearSystem == nullptr) {
            throw runtime_error("LinearSystem is not set");
        }
        return _linearSystem;
    }
    
    shared_ptr<MatrixDecomposition> DirectSolver::getDecomposition() {
        return nullptr;
    }
    
    void DirectSolver::_initializeVectors() {
        if (!_isLinearSystemSet)
            throw std::invalid_argument("Linear system must be set before setting the initial solution.");
        if (_linearSystem->solution == nullptr) {
            _linearSystem->solution = make_shared<vector<double>>(_linearSystem->rhs->size());
        }
        else
            throw runtime_error("Solution vector is already illegally initialized.");
        _vectorsInitialized = true;
    }

    void DirectSolver::setInitialSolution(shared_ptr<vector<double>> initialValue) {
        if (!_vectorsInitialized)
            throw runtime_error("Vectors must be initialized before setting the initial solution.");
        if (initialValue->size() != _linearSystem->solution->size())
            throw std::invalid_argument("Initial value vector must have the same size as the solution vector.");
        _linearSystem->solution = std::move(initialValue);
        _solutionSet = true;   
    }
    
    void DirectSolver::setInitialSolution(double initialValue) {
        if (!_vectorsInitialized)
            throw runtime_error("Vectors must be initialized before setting the initial solution.");
        for (auto &value : *_linearSystem->solution) {
            value = initialValue;
        }
        _solutionSet = true;
    }


} // LinearAlgebra