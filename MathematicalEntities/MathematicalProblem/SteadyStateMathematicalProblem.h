//
// Created by hal9000 on 12/17/22.
//
#ifndef UNUNTITLED_STEADYSTATEMATHEMATICALPROBLEM_H
#define UNUNTITLED_STEADYSTATEMATHEMATICALPROBLEM_H
#include "MathematicalProblem.h"
#include "../PartialDifferentialEquations/SteadyStatePartialDifferentialEquation.h"
namespace MathematicalEntities{
    class SteadyStateMathematicalProblem : public MathematicalProblem{
    public:
        SteadyStateMathematicalProblem(shared_ptr<SteadyStatePartialDifferentialEquation> pde,
                                       shared_ptr<DomainBoundaryConditions> bcs,
                                       struct Field_DOFType* degreesOfFreedom);
        
        shared_ptr<SteadyStatePartialDifferentialEquation> steadyStatePDE;
    };
    
}

#endif //UNUNTITLED_STEADYSTATEMATHEMATICALPROBLEM_H