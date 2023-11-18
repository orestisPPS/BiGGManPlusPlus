//
// Created by hal9000 on 12/17/22.
//

#include "TransientMathematicalProblem.h"

#include <utility>

namespace MathematicalEntities {
    
    TransientMathematicalProblem::TransientMathematicalProblem(
            shared_ptr<PartialDifferentialEquation> pde, shared_ptr<DomainBoundaryConditions> bcs,
            shared_ptr<InitialConditions> initialConditions, struct Field_DOFType *degreesOfFreedom) :
            SteadyStateMathematicalProblem(std::move(pde), std::move(bcs), degreesOfFreedom),
            initialConditions(std::move(initialConditions)) {
        
    }


}// MathematicalEntities