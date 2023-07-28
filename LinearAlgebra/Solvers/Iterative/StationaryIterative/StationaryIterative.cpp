//
// Created by hal9000 on 6/14/23.
//

#include "StationaryIterative.h"

namespace LinearAlgebra {
    StationaryIterative::StationaryIterative(VectorNormType normType, double tolerance, unsigned maxIterations, bool throwExceptionOnMaxFailure, ParallelizationMethod parallelizationMethod) :
            IterativeSolver(normType, tolerance, maxIterations, throwExceptionOnMaxFailure, parallelizationMethod) {
    }

    void StationaryIterative::_iterativeSolution() {
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

            _stationaryIterativeCuda = make_unique<StationaryIterativeCuda>(
                    d_matrix, d_rhs, d_xOld, d_xNew, d_difference, vectorSize, 256);

            while (iteration < _maxIterations && norm >= _tolerance) {

                _stationaryIterativeCuda->performGaussSeidelIteration();
                _stationaryIterativeCuda->getDifferenceVector(_difference->data());
                norm = VectorNorm(_difference, _normType).value();
                //norm = _stationaryIterativeCuda->getNorm();
                // Add the norm to the list of norms
                _residualNorms->push_back(norm);

                if (iteration % 100 == 0) {
                    cout << "Iteration: " << iteration << " - Norm: " << norm << endl;
                }

                iteration++;
            }
            _stationaryIterativeCuda->getSolutionVector(_xNew->data());
            _stationaryIterativeCuda.reset();

        }

        auto end = std::chrono::high_resolution_clock::now();
        printAnalysisOutcome(iteration, norm, start, end);
        
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