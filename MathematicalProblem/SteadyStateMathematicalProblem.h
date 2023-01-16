//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "../BoundaryConditions/BoundaryCondition.h"

namespace MathematicalProblem{
    class SteadyStateMathematicalProblem {
    public:
        SteadyStateMathematicalProblem(PartialDifferentialEquation *pde,
                                       map<Position,list<BoundaryConditions::BoundaryCondition*>> *bcs,
                                       list<DegreeOfFreedom*> *dof,
                                       PhysicalSpaceEntity *space);
        ~SteadyStateMathematicalProblem();
        PartialDifferentialEquation *pde;
        map<Position,list<BoundaryConditions::BoundaryCondition*>> *boundaryConditions;
        list<DegreeOfFreedom*> *degreesOfFreedom;
        PhysicalSpaceEntity *space;
    };
}

