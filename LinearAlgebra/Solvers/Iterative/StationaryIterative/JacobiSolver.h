
//
// Created by hal9000 on 6/13/23.
//

#ifndef UNTITLED_JACOBISOLVER_H
#define UNTITLED_JACOBISOLVER_H

#include "StationaryIterative.h"

namespace LinearAlgebra {

    class JacobiSolver : public StationaryIterative{
        
    public:
        JacobiSolver(ParallelizationMethod parallelizationMethod, VectorNormType normType, double tolerance = 1E-9, unsigned maxIterations = 1E4, bool throwExceptionOnMaxFailure = true);
    
    protected:
        void _threadJob(unsigned start, unsigned end) override;
    };

} // LinearAlgebra

#endif //UNTITLED_JACOBISOLVER_H