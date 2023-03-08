//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../BoundaryConditions/BoundaryCondition.h"

namespace MathematicalProblem {

    class TransientMathematicalProblem {
    public:
        TransientMathematicalProblem(PartialDifferentialEquation *pde,
                                     map<Position,list<BoundaryConditions::BoundaryCondition*>> *bcs,
                                     map<int*,double>* domainIC,
                                     struct Field_DOFType *degreesOfFreedom);
                
        PartialDifferentialEquation *pde;
        map<Position,list<BoundaryConditions::BoundaryCondition*>> *boundaryConditions;
        double* initialCondition;
        map<int*,double>* domainInitialConditions;
        struct Field_DOFType *degreesOfFreedom
    };

};
