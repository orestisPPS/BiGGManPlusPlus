//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_SORSOLVER_H
#define UNTITLED_SORSOLVER_H

#include "StationaryIterative.h"

namespace LinearAlgebra{
    class SORSolver : public StationaryIterative {

    public:
        SORSolver(double relaxationParameter, VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4,
                  bool throwExceptionOnMaxFailure = true, ParallelizationMethod parallelizationMethod = Wank);

    protected:
        void _singleThreadSolution() override;

        void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows) override;
    
    private:
        void _threadJobSOR(unsigned start, unsigned end);

        double _relaxationParameter;
    };
}


#endif //UNTITLED_SORSOLVER_H
