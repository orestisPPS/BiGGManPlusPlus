//
// Created by hal9000 on 6/14/23.
//

#include "StationaryIterative.h"

namespace LinearAlgebra {
    StationaryIterative::StationaryIterative(double tolerance, unsigned maxIterations, VectorNormType normType,
                                             unsigned userDefinedThreads, bool printOutput, bool throwExceptionOnMaxFailure) :
            IterativeSolver(tolerance, maxIterations, normType, userDefinedThreads, printOutput, throwExceptionOnMaxFailure) {
        _difference = nullptr;
    }


    
    void StationaryIterative::_initializeVectors() {
        _xNew = make_shared<NumericalVector<double>>(_linearSystem->rhs->size());
        _xOld = make_shared<NumericalVector<double>>(_linearSystem->rhs->size());
        _difference = make_shared<NumericalVector<double>>(_linearSystem->rhs->size());
        _vectorsInitialized = true;    
    }
    
    
    
    void StationaryIterative::_iterativeSolution() {
        auto start = std::chrono::high_resolution_clock::now();
        unsigned n = _linearSystem->matrix->numberOfRows();
        _exitNorm = 1.0;
        _difference = make_shared<NumericalVector<double>>(_linearSystem->rhs->size());

        while (_iteration < _maxIterations && _exitNorm >= _tolerance) {
            _performMethodIteration();
            _exitNorm = _xNew->norm(_normType);
            _printIterationAndNorm();
        }
            auto end = std::chrono::high_resolution_clock::now();
            printAnalysisOutcome(_iteration, _exitNorm, start, end);



/*        else if (_parallelization == CUDA) {
            
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

        }*/


        
    }
    

    void StationaryIterative::_cudaSolution() {
    }

    void StationaryIterative::_singleThreadSolution() {
    }
    


}
// LinearAlgebra

