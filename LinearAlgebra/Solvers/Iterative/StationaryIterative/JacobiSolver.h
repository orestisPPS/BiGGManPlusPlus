
//
// Created by hal9000 on 6/13/23.
//

#ifndef UNTITLED_JACOBISOLVER_H
#define UNTITLED_JACOBISOLVER_H

#include "StationaryIterative.h"

namespace LinearAlgebra {

    class JacobiSolver : public StationaryIterative{
        
    public:
        explicit JacobiSolver(VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4,
                     bool throwExceptionOnMaxFailure = true, ParallelizationMethod parallelizationMethod = Wank);
    protected:
        void _singleThreadSolution() override;

        void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows) override;
        
    private:
        void _threadJobJacobi(unsigned start, unsigned end);
        
    };

} // LinearAlgebra

#endif //UNTITLED_JACOBISOLVER_H