//
// Created by hal9000 on 12/17/22.
//

#include "TransientMathematicalProblem.h"

namespace MathematicalProblems {
    
    
    TransientMathematicalProblem :: TransientMathematicalProblem(
            shared_ptr<PartialDifferentialEquation>pde,
            shared_ptr<DomainBoundaryConditions> bcs, map<int*,double>* domainIC,
            struct Field_DOFType *degreesOfFreedom) :
            pde(pde), boundaryConditions(bcs), initialCondition(nullptr), domainInitialConditions(domainIC), degreesOfFreedom(degreesOfFreedom),
            MathematicalProblem(pde, bcs, degreesOfFreedom){
        
    }



}// MathematicalProblems