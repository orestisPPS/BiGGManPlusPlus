//
// Created by hal9000 on 12/17/22.
//
#pragma once
#include "MathematicalProblem.h"

namespace MathematicalProblem{
    class SteadyStateMathematicalProblem : public MathematicalProblem{
    public:
        SteadyStateMathematicalProblem(PartialDifferentialEquation* pde,
                                       DomainBoundaryConditions* bcs,
                                       struct Field_DOFType* degreesOfFreedom) ;

        ~SteadyStateMathematicalProblem();
        PartialDifferentialEquation* pde;
        DomainBoundaryConditions* boundaryConditions;
        Field_DOFType *degreesOfFreedom;
    };
    
}

