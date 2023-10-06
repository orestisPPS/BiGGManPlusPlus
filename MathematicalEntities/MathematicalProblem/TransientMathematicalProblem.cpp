//
// Created by hal9000 on 12/17/22.
//

#include "TransientMathematicalProblem.h"

#include <utility>

namespace MathematicalEntities {
    
    TransientMathematicalProblem::TransientMathematicalProblem(
            shared_ptr<PartialDifferentialEquation> pde, shared_ptr<DomainBoundaryConditions> bcs,
            double initialCondition, struct Field_DOFType *degreesOfFreedom) :
            SteadyStateMathematicalProblem(std::move(pde), std::move(bcs), degreesOfFreedom),
            _initialConditionOrderZero(initialCondition) {
        _domainInitialConditionsOrderZero = nullptr;
        _domainInitialConditionsOrderOne = nullptr;
    }
    
    TransientMathematicalProblem :: TransientMathematicalProblem(
            shared_ptr<PartialDifferentialEquation> pde, shared_ptr<DomainBoundaryConditions> bcs,
            shared_ptr<map<unsigned, double>> domainInitialConditions, struct Field_DOFType *degreesOfFreedom) :
            SteadyStateMathematicalProblem(std::move(pde), std::move(bcs), degreesOfFreedom),
            _domainInitialConditions(std::move(domainInitialConditions)) {
        _initialCondition = 0;
    }



}// MathematicalEntities