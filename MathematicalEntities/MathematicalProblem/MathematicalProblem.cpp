//
// Created by hal9000 on 3/13/23.
//

#include "MathematicalProblem.h"
using namespace MathematicalEntities;

namespace MathematicalEntities {
    
    MathematicalProblem::MathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                             shared_ptr<DomainBoundaryConditions> bcs,
                                             struct Field_DOFType* degreesOfFreedom) :
                                                     pde(std::move(pde)), boundaryConditions(std::move(bcs)), degreesOfFreedom(degreesOfFreedom) {
    }

} // MathematicalEntities