//
// Created by hal9000 on 12/17/22.
//
#ifndef UNUNTITLED_STEADYSTATEMATHEMATICALPROBLEM_H
#define UNUNTITLED_STEADYSTATEMATHEMATICALPROBLEM_H
#include "MathematicalProblem.h"
namespace MathematicalEntities{
    class SteadyStateMathematicalProblem : public MathematicalProblem{
    public:
        SteadyStateMathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                       shared_ptr<DomainBoundaryConditions> bcs,
                                       struct Field_DOFType* degreesOfFreedom);
        
    };
    
}

#endif //UNUNTITLED_STEADYSTATEMATHEMATICALPROBLEM_H