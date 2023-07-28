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
            _throwExceptionOnMaxFailure(throwExceptionOnMaxFailure), _isInitialized(false), _xNew(nullptr),
            _xOld(nullptr),
            _residualNorms(make_shared<list<double>>()) {
        _parallelization = parallelizationMethod;
    }

    IterativeSolver::~IterativeSolver() {
        _xNew.reset();
        _xOld.reset();
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

    void IterativeSolver::setLinearSystem(shared_ptr<LinearSystem> linearSystem) {
        _linearSystem = linearSystem;
        _isLinearSystemSet = true;
    }

    void IterativeSolver::setInitialSolution(shared_ptr<vector<double>> initialValue) {
        if (!_isLinearSystemSet)
            throw std::invalid_argument("Linear system must be set before setting the initial solution.");
        if (initialValue->size() != _linearSystem->rhs->size()) {
            throw std::invalid_argument("Initial solution vector must have the same size as the rhs vector.");
        }
        _xOld = std::move(initialValue);
        _xNew = make_unique<vector<double>>(_xOld->size(), 0.0);
        _difference = make_shared<vector<double>>(_xOld->size(), 0.0);
        _isInitialized = true;
    }

    void IterativeSolver::setInitialSolution(double initialValue) {
        if (!_isLinearSystemSet)
            throw std::invalid_argument("Linear system must be set before setting the initial solution.");
        _xOld = make_unique<vector<double>>(_linearSystem->rhs->size(), initialValue);
        _xNew = make_unique<vector<double>>(_xOld->size(), 0.0);
        _difference = make_shared<vector<double>>(_xOld->size(), 0.0);
        _isInitialized = true;
    }

    void IterativeSolver::solve() {
        if (!_isLinearSystemSet)
            throw std::invalid_argument("Linear system must be set before solving.");
        _iterativeSolution();
        _linearSystem->solution = std::move(_xNew);
    }

    void IterativeSolver::_iterativeSolution() {

        auto start = std::chrono::high_resolution_clock::now();
        unsigned n = _linearSystem->matrix->numberOfRows();
        unsigned short iteration = 0;
        double norm = 1.0;

        if (!_isInitialized)
            throw std::invalid_argument("Initial solution must be set before solving.");
        switch (_parallelization) {
            case Wank :

                break;
            case vTechKickInYoo : {

                break;
            }
            case turboVTechKickInYoo:
                auto lol = 1;
                _cudaSolution();
                break;
        }
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

    void IterativeSolver::_printIterationAndNorm(unsigned int iteration, double norm) {
        if (iteration % 100 == 0)
            cout << "Iteration: " << iteration << " - Norm: " << norm << endl;

    }

    double IterativeSolver::_calculateNorm() {
        double norm = VectorNorm(_difference, _normType).value();
        _residualNorms->push_back(norm);
        return norm;
    }

    void IterativeSolver::printAnalysisOutcome(unsigned totalIterations, double exitNorm,  std::chrono::high_resolution_clock::time_point startTime,
                                                   std::chrono::high_resolution_clock::time_point finishTime){
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