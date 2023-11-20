//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_GAUSSSEIDELSOLVER_H
#define UNTITLED_GAUSSSEIDELSOLVER_H

#include "StationaryIterative.h"
namespace LinearAlgebra {

    class GaussSeidelSolver : public StationaryIterative{

    public:
        explicit GaussSeidelSolver(double tolerance = 1E-5, unsigned maxIterations = 1E4, VectorNormType normType = L2,
                                   unsigned userDefinedThreads = 0, bool printOutput = true, bool throwExceptionOnMaxFailure = true);

    protected:
        
        void _performMethodSolution() override;
        
    private:
    };  
} // LinearAlgebra

#endif //UNTITLED_GAUSSSEIDELSOLVER_H
