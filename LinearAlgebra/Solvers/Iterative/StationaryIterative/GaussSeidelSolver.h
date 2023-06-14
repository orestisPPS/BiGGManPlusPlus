//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_GAUSSSEIDELSOLVER_H
#define UNTITLED_GAUSSSEIDELSOLVER_H

#include "StationaryIterative.h"

namespace LinearAlgebra {

    class GaussSeidelSolver : public StationaryIterative{

    public:
        GaussSeidelSolver(bool vTechKickInYoo, VectorNormType normType, double tolerance = 1E-9, unsigned maxIterations = 1E4, bool throwExceptionOnMaxFailure = true);

    protected:
        void _threadJob(unsigned start, unsigned end) override;
    };  
} // LinearAlgebra

#endif //UNTITLED_GAUSSSEIDELSOLVER_H
