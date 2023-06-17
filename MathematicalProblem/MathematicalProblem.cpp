//
// Created by hal9000 on 3/13/23.
//

#include "MathematicalProblem.h"
using namespace MathematicalProblems;

namespace MathematicalProblems {
    
    MathematicalProblem::MathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                             shared_ptr<DomainBoundaryConditions> bcs,
                                             struct Field_DOFType* degreesOfFreedom) :
                                                     pde(pde), boundaryConditions(bcs), degreesOfFreedom(degreesOfFreedom) {
    }
    
} // MathematicalProblems