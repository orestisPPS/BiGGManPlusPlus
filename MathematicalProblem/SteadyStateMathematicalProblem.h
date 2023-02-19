//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../BoundaryConditions/DomainBoundaryConditions.h"

namespace MathematicalProblem{
    class SteadyStateMathematicalProblem {
    public:
        SteadyStateMathematicalProblem(PartialDifferentialEquation* pde,
                                       DomainBoundaryConditions* bcs,
                                       list<DegreeOfFreedom* >* dof, SpaceEntityType space);

        ~SteadyStateMathematicalProblem();
        PartialDifferentialEquation* pde;
        DomainBoundaryConditions* boundaryConditions;
        list<DegreeOfFreedom*>* degreesOfFreedom;
        SpaceEntityType space;
    
    private:
        void checkDegreesOfFreedom() const;
        void checkSpaceEntityType() const;
    };
    
}

