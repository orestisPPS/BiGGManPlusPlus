//
// Created by hal9000 on 12/17/22.
//

#include "SteadyStateMathematicalProblem.h"
namespace MathematicalEntities {

    SteadyStateMathematicalProblem::SteadyStateMathematicalProblem(
            shared_ptr<PartialDifferentialEquation>pde, shared_ptr<DomainBoundaryConditions>bcs, struct Field_DOFType *degreesOfFreedom) :
            pde(std::move(pde)), boundaryConditions(std::move(bcs)), degreesOfFreedom(degreesOfFreedom),
            MathematicalProblem(pde, bcs, degreesOfFreedom){
        
    }
}
