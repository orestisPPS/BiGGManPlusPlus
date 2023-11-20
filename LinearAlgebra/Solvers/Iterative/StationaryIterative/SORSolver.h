//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_SORSOLVER_H
#define UNTITLED_SORSOLVER_H

#include "StationaryIterative.h"

namespace LinearAlgebra{
    class SORSolver : public StationaryIterative {

    public:
        explicit SORSolver(double relaxationParameter, double tolerance = 1E-5, unsigned maxIterations = 1E4, VectorNormType normType = L2,
                           unsigned userDefinedThreads = 0, bool printOutput = true, bool throwExceptionOnMaxFailure = true);

    protected:
        void _performMethodSolution() override;
        
    private:
        
        double _relaxationParameter;
    };
}


#endif //UNTITLED_SORSOLVER_H
