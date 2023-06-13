
//
// Created by hal9000 on 6/13/23.
//

#ifndef UNTITLED_JACOBISOLVER_H
#define UNTITLED_JACOBISOLVER_H

#include "IterativeSolver.h"

namespace LinearAlgebra {

    class JacobiSolver : public IterativeSolver {
        
    public:
        JacobiSolver(bool vTechKickInYoo, VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4, bool throwExceptionOnMaxFailure = true);
    
    protected:
        void _iterativeSolution() override;

    private:
        void multiThreadSolution();

        void singleThreadSolution();

        bool _vTechKickInYoo;
    };

} // LinearAlgebra

#endif //UNTITLED_JACOBISOLVER_H