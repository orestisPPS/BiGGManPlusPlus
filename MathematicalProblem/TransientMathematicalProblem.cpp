//
// Created by hal9000 on 12/17/22.
//

#include "TransientMathematicalProblem.h"

namespace MathematicalProblem {
    TransientMathematicalProblem::TransientMathematicalProblem(PartialDifferentialEquation *pde,
                                                               map<Position, list<BoundaryConditions::BoundaryCondition *>> *bcs,
                                                               double *ic,
                                                               list<DegreeOfFreedom *> *dof,
                                                               CoordinateSystem coordinateSystem) {
        pde = pde;
        boundaryConditions = bcs;
        initialCondition = ic;
        domainInitialConditions = nullptr;
        degreesOfFreedom = dof;
        coordinateSystem = coordinateSystem;
    }
    
    TransientMathematicalProblem::TransientMathematicalProblem(PartialDifferentialEquation *pde,
                                                               map<Position, list<BoundaryConditions::BoundaryCondition *>> *bcs,
                                                               map<int *, double> *domainIC,
                                                               list<DegreeOfFreedom *> *dof,
                                                               CoordinateSystem coordinateSystem) {
        pde = pde;
        boundaryConditions = bcs;
        initialCondition = nullptr;
        domainInitialConditions = domainIC;
        degreesOfFreedom = dof;
        coordinateSystem = coordinateSystem;
    }
    
    TransientMathematicalProblem::~TransientMathematicalProblem() {
        delete pde;
        delete boundaryConditions;
        delete domainInitialConditions;
        delete degreesOfFreedom;
    }

}// MathematicalProblem