//
// Created by hal9000 on 12/17/22.
//

#include "SteadyStateMathematicalProblem.h"
namespace MathematicalProblems {

    SteadyStateMathematicalProblem::SteadyStateMathematicalProblem(
            PartialDifferentialEquation *pde, DomainBoundaryConditions *bcs, struct Field_DOFType *degreesOfFreedom) :
            pde(pde), boundaryConditions(bcs), degreesOfFreedom(degreesOfFreedom),
            MathematicalProblem(pde, bcs, degreesOfFreedom){
        
    }
    
    SteadyStateMathematicalProblem::~SteadyStateMathematicalProblem() {
        delete pde;
        delete boundaryConditions;
        pde = nullptr;
        boundaryConditions = nullptr;
        degreesOfFreedom = nullptr;
    }
    
}
