//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../BoundaryConditions/BoundaryCondition.h"
#include "MathematicalProblem.h"

namespace MathematicalProblems {

    class TransientMathematicalProblem : public MathematicalProblem{
        
    public:
        
        TransientMathematicalProblem(shared_ptr<PartialDifferentialEquation>pde,
                                     shared_ptr<DomainBoundaryConditions> bcs, map<int*,double>* domainIC,
                                     struct Field_DOFType *degreesOfFreedom);
        
        shared_ptr<PartialDifferentialEquation>pde;
        
        shared_ptr<DomainBoundaryConditions> boundaryConditions;
        
        double* initialCondition;
        
        map<int*,double>* domainInitialConditions;
        
        struct Field_DOFType *degreesOfFreedom;
    };

};
