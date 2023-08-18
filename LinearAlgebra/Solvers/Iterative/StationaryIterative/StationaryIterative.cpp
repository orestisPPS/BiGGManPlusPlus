//
// Created by hal9000 on 6/14/23.
//

#include "StationaryIterative.h"

namespace LinearAlgebra {
    StationaryIterative::StationaryIterative(VectorNormType normType, double tolerance, unsigned maxIterations, bool throwExceptionOnMaxFailure, ParallelizationMethod parallelizationMethod) :
            IterativeSolver(normType, tolerance, maxIterations, throwExceptionOnMaxFailure, parallelizationMethod) {
        _difference = nullptr;
    }


    
    void StationaryIterative::_initializeVectors() {
        _xNew = make_shared<vector<double>>(_linearSystem->rhs->size());
        _xOld = make_shared<vector<double>>(_linearSystem->rhs->size());
        _difference = make_shared<vector<double>>(_linearSystem->rhs->size());
        _vectorsInitialized = true;    
    }
    
    
    
    void StationaryIterative::_iterativeSolution() {
        auto start = std::chrono::high_resolution_clock::now();
        unsigned n = _linearSystem->matrix->numberOfRows();
        _exitNorm = 1.0;
        _difference = make_shared<vector<double>>(n);
        
        if (_parallelization == SingleThread) {
            _printSingleThreadInitializationText();
            while (_iteration < _maxIterations && _exitNorm >= _tolerance) {
                _singleThreadSolution();
                _exitNorm = _calculateNorm();
                _printIterationAndNorm();
                _iteration++;
            }
        }
        if (_parallelization == MultiThread) {
            auto numberOfThreads = std::thread::hardware_concurrency();
            _printMultiThreadInitializationText(numberOfThreads);
            while (_iteration < _maxIterations && _exitNorm >= _tolerance) {
                _multiThreadSolution(numberOfThreads, n);
                _exitNorm = _calculateNorm();
                _printIterationAndNorm();
                _iteration++;
            }

        }

        else if (_parallelization == CUDA) {
            
            double *d_matrix = _linearSystem->matrix->getArrayPointer();
            double *d_rhs = _linearSystem->rhs->data();
            double *d_xOld = _xOld->data();
            double *d_xNew = _xNew->data();
            double *d_difference = _difference->data();
            int vectorSize = static_cast<int>(_linearSystem->rhs->size());

            _stationaryIterativeCuda = make_unique<StationaryIterativeCuda>(
                    d_matrix, d_rhs, d_xOld, d_xNew, d_difference, vectorSize, 256);

            while (_iteration < _maxIterations && _exitNorm >= _tolerance) {

                _stationaryIterativeCuda->performGaussSeidelIteration();
                _stationaryIterativeCuda->getDifferenceVector(_difference->data());
                _exitNorm = VectorNorm(_difference, _normType).value();
                //norm = _stationaryIterativeCuda->getNorm();
                // Add the norm to the list of norms
                _residualNorms->push_back(_exitNorm);

                if (_iteration % 100 == 0) {
                    cout << "Iteration: " << _iteration << " - Norm: " << _exitNorm << endl;
                }

                _iteration++;
            }
            _stationaryIterativeCuda->getSolutionVector(_xNew->data());
            _stationaryIterativeCuda.reset();

        }

        auto end = std::chrono::high_resolution_clock::now();
        printAnalysisOutcome(_iteration, _exitNorm, start, end);
        
    }

    void StationaryIterative::_multiThreadSolution(const unsigned short &availableThreads,
                                                   const unsigned short &numberOfRows) {
    }

    void StationaryIterative::_cudaSolution() {
    }

    void StationaryIterative::_singleThreadSolution() {
    }
    


}
// LinearAlgebra

