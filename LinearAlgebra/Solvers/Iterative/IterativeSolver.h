//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_ITERATIVESOLVER_H
#define UNTITLED_ITERATIVESOLVER_H

#include "../Solver.h"
#include "../../LinearSystem.h"
#include "../../Norms/VectorNorm.h"

namespace LinearAlgebra {

    class IterativeSolver : public Solver {
    public:
        IterativeSolver(LinearSystem* linearSystem,
                        VectorNormType normType, double tolerance, unsigned maxIterations);

        ~IterativeSolver();
        
        LinearSystem* linearSystem;
        
        VectorNormType normType;
        
        double tolerance;
        
        unsigned maxIterations;
        
        
    protected:
        

    };

} // LinearAlgebra

#endif //UNTITLED_ITERATIVESOLVER_H
