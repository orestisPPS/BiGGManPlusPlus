//
// Created by hal9000 on 7/27/23.
//

#include "ConjugateGradientSolver.h"

namespace LinearAlgebra {
    ConjugateGradientSolver::ConjugateGradientSolver(ParallelizationMethod parallelizationMethod,
                                                     VectorNormType normType, double tolerance,
                                                     unsigned int maxIterations, bool throwExceptionOnMaxFailure) :
                                                     IterativeSolver(normType, tolerance, maxIterations, throwExceptionOnMaxFailure) {
        
    }

    void ConjugateGradientSolver::_iterativeSolution() {
        auto start = std::chrono::high_resolution_clock::now();
        unsigned n = _linearSystem->matrix->numberOfRows();
        unsigned short iteration = 0;
        double norm = 1.0;

        //Check if the initial solution has been set from the user. If not, it is initialized to 0.0
        if (!_isInitialized)
            setInitialSolution(0.0);

        if (_parallelization == Wank) {
            _printSingleThreadInitializationText();
            while (iteration < _maxIterations && norm >= _tolerance) {
                _singleThreadSolution();
                norm = _calculateNorm();
                _printIterationAndNorm(iteration, norm);
                iteration++;
            }
        }
        if (_parallelization == vTechKickInYoo) {
            auto numberOfThreads = std::thread::hardware_concurrency();
            _printMultiThreadInitializationText(numberOfThreads);
            while (iteration < _maxIterations && norm >= _tolerance) {
                _multiThreadSolution(numberOfThreads, n);
                norm = _calculateNorm();
                _printIterationAndNorm(iteration, norm);
                iteration++;
            }

        }

        else if (_parallelization == turboVTechKickInYoo) {

            double *d_matrix = _linearSystem->matrix->getArrayPointer();
            double *d_rhs = _linearSystem->rhs->data();
            double *d_xOld = _xOld->data();
            double *d_xNew = _xNew->data();
            double *d_difference = _difference->data();
            int vectorSize = static_cast<int>(_linearSystem->rhs->size());

    

        }

        auto end = std::chrono::high_resolution_clock::now();
        printAnalysisOutcome(iteration, norm, start, end);
    }

    void ConjugateGradientSolver::_multiThreadSolution(const unsigned short &availableThreads,
                                                       const unsigned short &numberOfRows) {
        
    }

    void ConjugateGradientSolver::_cudaSolution() {
        IterativeSolver::_cudaSolution();
    }

    void ConjugateGradientSolver::_singleThreadSolution() {
        _printSingleThreadInitializationText();
        auto start = std::chrono::high_resolution_clock::now();
        unsigned n = _linearSystem->matrix->numberOfRows();
        unsigned short iteration = 0;
        double norm = 1.0;
        
        //Calculate the initial residual
        _r = VectorOperations::subtract(_linearSystem->rhs, _linearSystem->matrix->multiplyWithVector(*_xOld));
        
        
        while (iteration < _maxIterations && norm >= _tolerance) {
            _singleThreadSolution();
            
            
            
            norm = _calculateNorm();
            _printIterationAndNorm(iteration, norm);
            iteration++;
        }
        auto end = std::chrono::high_resolution_clock::now();
        printAnalysisOutcome(iteration, norm, start, end);
        
    }


} // LinearAlgebra