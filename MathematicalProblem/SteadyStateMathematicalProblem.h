//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../BoundaryConditions/BoundaryConditionSet.h"

namespace MathematicalProblem{
    class SteadyStateMathematicalProblem {
    public:
        SteadyStateMathematicalProblem(PartialDifferentialEquation* pde,
                                       map<Position, list<BoundaryConditions::BoundaryCondition* >* >* bcs,
                                       list<DegreeOfFreedom* >* dof, SpaceEntityType space);

        ~SteadyStateMathematicalProblem();
        PartialDifferentialEquation* pde;
        map<Position, list<BoundaryCondition* >* >* boundaryConditions;
        list<DegreeOfFreedom*>* degreesOfFreedom;
        SpaceEntityType space;
    
    private:
        void checkBoundaryConditions() const;
        void checkDegreesOfFreedom() const;
        void checkSpaceEntityType() const;
    };
    
}

