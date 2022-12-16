//
// Created by hal9000 on 12/17/22.
//

#include "SteadyStateMathematicalProblem.h"
namespace MathematicalProblem{
    
    SteadyStateMathematicalProblem::SteadyStateMathematicalProblem(PartialDifferentialEquation *pde,
                                                                   map<Position, list<BoundaryConditions::BoundaryCondition *>> *bcs,
                                                                   list<DegreeOfFreedom *> *dof, CoordinateSystem coordinateSystem) {
        pde = pde;
        boundaryConditions = bcs;
        degreesOfFreedom = dof;
        coordinateSystem = coordinateSystem;
    }

    SteadyStateMathematicalProblem::~SteadyStateMathematicalProblem() {
        delete pde;
        delete boundaryConditions;
        delete degreesOfFreedom;
    }
}
