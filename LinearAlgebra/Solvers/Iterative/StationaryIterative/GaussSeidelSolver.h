//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_GAUSSSEIDELSOLVER_H
#define UNTITLED_GAUSSSEIDELSOLVER_H

#include "StationaryIterative.h"
namespace LinearAlgebra {

    class GaussSeidelSolver : public StationaryIterative{

    public:
        explicit GaussSeidelSolver(VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4,
                          bool throwExceptionOnMaxFailure = true, ParallelizationMethod parallelizationMethod = SingleThread);

    protected:
        
        void _singleThreadSolution() override;
        
        void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows) override;

    private:
        void _threadJobGaussSeidel(unsigned start, unsigned end);
    };  
} // LinearAlgebra

#endif //UNTITLED_GAUSSSEIDELSOLVER_H
