//
// Created by hal9000 on 4/18/23.
//

#include "IterativeSolver.h"

#include <memory>

namespace LinearAlgebra {

/*
    auto x = VectorNorm::_calculateLInfNorm(new NumericalVector<double>());
*/

    IterativeSolver::IterativeSolver(double tolerance, unsigned maxIterations, VectorNormType normType,
                                     unsigned userDefinedThreads, bool printOutput, bool throwExceptionOnMaxFailure) :
        Solver(),
        _tolerance(tolerance), _maxIterations(maxIterations), _normType(normType), _userDefinedThreads(userDefinedThreads),
        _printOutput(printOutput), _throwExceptionOnMaxFailure(throwExceptionOnMaxFailure), _iteration(0), _exitNorm(0.0),
        _xNew(nullptr), _xOld(nullptr), _residualNorms(make_shared<list<double>>()) {
        
        _linearSystemInitialized = false;
        _vectorsInitialized = false;
    }

    IterativeSolver::~IterativeSolver() {
        _xNew.reset();
        _xOld.reset();
    }
    
    void IterativeSolver::setInitialSolution(shared_ptr<NumericalVector<double>> initialSolution) {
        if (!_vectorsInitialized)
            throw runtime_error("Vectors must be initialized before setting initial solution.");
        if (initialSolution->size() != _linearSystem->solution->size())
            throw std::invalid_argument("Initial solution vector must have the same size as the solution vector.");
        _xOld = std::move(initialSolution);
        _solutionSet = true;
    }
    
    void IterativeSolver::setInitialSolution(double initialSolution) {
        if (!_vectorsInitialized)
            throw runtime_error("Vectors must be initialized before setting initial solution.");
        for (auto &value : *_xOld) {
            value = initialSolution;
        }
        _solutionSet = true;
    }

    void IterativeSolver::setTolerance(double tolerance) {
        _tolerance = tolerance;
    }

    const double &IterativeSolver::getTolerance() const {
        return _tolerance;
    }


    void IterativeSolver::setMaxIterations(unsigned maxIterations) {
        _maxIterations = maxIterations;
    }

    const unsigned &IterativeSolver::getMaxIterations() const {
        return _maxIterations;
    }

    void IterativeSolver::setNormType(VectorNormType normType) {
        _normType = normType;
    }

    const VectorNormType &IterativeSolver::getNormType() const {
        return _normType;
    }

    void IterativeSolver::solve() {
        if (!_isLinearSystemSet)
            throw std::invalid_argument("Linear system must be set before solving.");
        _iterativeSolution();
        _linearSystem->solution = std::move(_xNew);
    }
    
    void IterativeSolver::_iterativeSolution() {
        
    }
    
    void IterativeSolver::_performMethodIteration() {
        
    }
    
    void IterativeSolver::_cudaSolution() {
        
    }

    void IterativeSolver::_printInitializationText() {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << _solverName << " Solver Initialized" << endl;
        unsigned matrixThreads = _linearSystem->matrix->dataStorage->getAvailableThreads();
        unsigned availableThreads = _userDefinedThreads != 0 ? _userDefinedThreads : matrixThreads;
        cout << "Threads Assigned for Solution: " << availableThreads << " out of " << thread::hardware_concurrency() << " available." << endl;
    }
    
    void IterativeSolver::_printCUDAInitializationText() {

    }

    void IterativeSolver::_printIterationAndNorm(unsigned displayFrequency) const {
        if (_iteration % displayFrequency == 0)
            cout << "Iteration: " << _iteration << " - Norm: " << _exitNorm << endl;

    }
    
    void IterativeSolver::printAnalysisOutcome(unsigned totalIterations, double exitNorm,  std::chrono::high_resolution_clock::time_point startTime,
                                                   std::chrono::high_resolution_clock::time_point finishTime) const{
        bool isInMicroSeconds = false;
        auto _elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();
        if (_elapsedTime == 0) {
            _elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(finishTime - startTime).count();
            isInMicroSeconds = true;
        }
        if (isInMicroSeconds) {
            if (exitNorm <= _tolerance)
                cout << "Convergence Achieved!" << endl;
            else
                cout << "Convergence Failed!" << endl;

            cout << "Elapsed time: " << _elapsedTime << " μs" << " Iterations : " << totalIterations << " Exit norm : " << exitNorm << endl;
            cout << "----------------------------------------" << endl;
        } else {

            if (exitNorm <= _tolerance)
                cout << "Convergence Achieved!" << endl;
            else
                cout << "Convergence Failed!" << endl;

            cout << "Elapsed time: " << _elapsedTime << " ms" << " Iterations : " << totalIterations << " Exit norm : " << exitNorm << endl;
            cout << "----------------------------------------" << endl;
        }
    }




} // LinearAlgebra