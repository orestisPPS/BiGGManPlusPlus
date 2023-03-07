//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../BoundaryConditions/DomainBoundaryConditions.h"
#include "../DegreesOfFreedom/DegreeOfFreedomTypes.h"

namespace MathematicalProblem{
    class SteadyStateMathematicalProblem {
    public:
        SteadyStateMathematicalProblem(PartialDifferentialEquation* pde,
                                       DomainBoundaryConditions* bcs,
                                       struct Field_DOFType* dofs);

        ~SteadyStateMathematicalProblem();
        PartialDifferentialEquation* pde;
        DomainBoundaryConditions* boundaryConditions;
        struct DegreesOfFreedom::Field_DOFType degreesOfFreedom;
        SpaceEntityType space;
    
    private:
        void checkDegreesOfFreedom() const;
        void checkSpaceEntityType() const;
    };
    
}

