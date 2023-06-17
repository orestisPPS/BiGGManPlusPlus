//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MathematicalProblem.h"

namespace MathematicalProblems{
    class SteadyStateMathematicalProblem : public MathematicalProblem{
    public:
        SteadyStateMathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                       shared_ptr<DomainBoundaryConditions> bcs,
                                       struct Field_DOFType* degreesOfFreedom) ;

        shared_ptr<PartialDifferentialEquation> pde;
        shared_ptr<DomainBoundaryConditions> boundaryConditions;
        Field_DOFType *degreesOfFreedom;
    };
    
}

