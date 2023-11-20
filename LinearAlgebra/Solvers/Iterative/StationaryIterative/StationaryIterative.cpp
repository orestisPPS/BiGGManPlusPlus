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
    
    void _performMethodIteration(){
    }
    

    

    void StationaryIterative::_cudaSolution() {
    }

    void StationaryIterative::_performMethodSolution() {
    }

}
// LinearAlgebra

