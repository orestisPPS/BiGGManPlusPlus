//
// Created by hal9000 on 12/17/22.
//

#include "SteadyStateMathematicalProblem.h"
namespace MathematicalProblem{
    
    SteadyStateMathematicalProblem::SteadyStateMathematicalProblem(PartialDifferentialEquation* pde,
                                                                   DomainBoundaryConditions* bcs,
                                                                   list<DegreeOfFreedom*>* dof,
                                                                   SpaceEntityType space)
    : pde(pde), boundaryConditions(bcs), degreesOfFreedom(dof), space(space){
        checkDegreesOfFreedom();
        checkSpaceEntityType();
    }

    SteadyStateMathematicalProblem::~SteadyStateMathematicalProblem() {
        delete pde;
        delete boundaryConditions;
        delete degreesOfFreedom;
        pde = nullptr;
        boundaryConditions = nullptr;
        degreesOfFreedom = nullptr;
    }
        
    void SteadyStateMathematicalProblem::checkDegreesOfFreedom() const {
        //TODO : Implement this method
    }
    
    void SteadyStateMathematicalProblem::checkSpaceEntityType() const {
        if (space == PositioningInSpace::NullSpace){
            throw invalid_argument("The void inside you cannot be solved. Like this PDE you try to solve on a"
                                   "null space.");
        }
    }
}
