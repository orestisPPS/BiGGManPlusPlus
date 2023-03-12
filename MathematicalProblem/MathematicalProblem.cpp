//
// Created by hal9000 on 3/13/23.
//

#include "MathematicalProblem.h"
using namespace MathematicalProblem;

namespace MathematicalProblem {
    
    MathematicalProblem::MathematicalProblem(PartialDifferentialEquation* pde,
                                             DomainBoundaryConditions* bcs,
                                             struct Field_DOFType* degreesOfFreedom) :
                                                     pde(pde), boundaryConditions(bcs), degreesOfFreedom(degreesOfFreedom) {
    }
    
    MathematicalProblem::~MathematicalProblem() {
        delete pde;
        delete boundaryConditions;
        pde = nullptr;
        boundaryConditions = nullptr;
        degreesOfFreedom = nullptr;
    }
} // MathematicalProblem