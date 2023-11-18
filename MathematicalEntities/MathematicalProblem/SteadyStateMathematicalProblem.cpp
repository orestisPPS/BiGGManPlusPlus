//
// Created by hal9000 on 12/17/22.
//

#include "SteadyStateMathematicalProblem.h"

namespace MathematicalEntities {

    SteadyStateMathematicalProblem::SteadyStateMathematicalProblem(
            shared_ptr<PartialDifferentialEquation> pde, shared_ptr<DomainBoundaryConditions> bcs, struct Field_DOFType *degreesOfFreedom) :
            MathematicalProblem(std::move(pde), std::move(bcs), degreesOfFreedom) {
        
    }
}
