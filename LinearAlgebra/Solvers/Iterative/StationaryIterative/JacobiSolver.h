
//
// Created by hal9000 on 6/13/23.
//

#ifndef UNTITLED_JACOBISOLVER_H
#define UNTITLED_JACOBISOLVER_H

#include "StationaryIterative.h"

namespace LinearAlgebra {

    class JacobiSolver : public StationaryIterative{
        
    public:
        explicit JacobiSolver(double tolerance = 1E-5, unsigned maxIterations = 1E4, VectorNormType normType = L2,
                              unsigned userDefinedThreads = 0, bool printOutput = true, bool throwExceptionOnMaxFailure = true);
    protected:
        void _performMethodIteration() override;
        
    };

} // LinearAlgebra

#endif //UNTITLED_JACOBISOLVER_H