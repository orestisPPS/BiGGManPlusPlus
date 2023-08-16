//
// Created by hal9000 on 4/18/23.
//

#include "IterativeSolver.h"

#include <memory>

namespace LinearAlgebra {

/*
    auto x = VectorNorm::_calculateLInfNorm(new vector<double>());
*/

    IterativeSolver::IterativeSolver(VectorNormType normType, double tolerance, unsigned maxIterations,
                                     bool throwExceptionOnMaxFailure, ParallelizationMethod parallelizationMethod) :
            Solver(), _normType(normType), _tolerance(tolerance), _maxIterations(maxIterations),
            _throwExceptionOnMaxFailure(throwExceptionOnMaxFailure), _xNew(nullptr),
            _xOld(nullptr),
            _residualNorms(make_shared<list<double>>()) {
        _linearSystemInitialized = false;
        _vectorsInitialized = false;
        _parallelization = parallelizationMethod;
        _iteration = 0;
    }

    IterativeSolver::~IterativeSolver() {
        _xNew.reset();
        _xOld.reset();
    }
    
    void IterativeSolver::setInitialSolution(shared_ptr<vector<double>> initialSolution) {
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
    
    void IterativeSolver::_singleThreadSolution() {
        
    }

    void IterativeSolver::_multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows) {

    }
    
    void IterativeSolver::_cudaSolution() {
        
    }

    void IterativeSolver::_printSingleThreadInitializationText() {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << _solverName << " Solver Single Thread - no vtec yo :(" << endl;
    }

    void IterativeSolver::_printMultiThreadInitializationText(unsigned short numberOfThreads) {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << _solverName << " Solver Multi Thread - VTEC KICKED IN YO!" << endl;
        //Find the number of threads available for parallel execution
        cout << "Total Number of threads available for parallel execution: " << numberOfThreads << endl;
        cout << "Number of threads involved in parallel solution: " << numberOfThreads << endl;
    }

    void IterativeSolver::_printCUDAInitializationText() {

    }

    void IterativeSolver::_printIterationAndNorm(unsigned displayFrequency) const {
        if (_iteration % displayFrequency == 0)
            cout << "Iteration: " << _iteration << " - Norm: " << _exitNorm << endl;

    }
    
    double IterativeSolver::_calculateNorm() {
        double norm = VectorNorm(_difference, _normType).value();
        _residualNorms->push_back(norm);
        return norm;
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