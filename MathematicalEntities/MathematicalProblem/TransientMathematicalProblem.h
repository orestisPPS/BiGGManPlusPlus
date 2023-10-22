//
// Created by hal9000 on 12/17/22.
//

#ifndef UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H
#define UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H

#include "../PartialDifferentialEquations/TransientPartialDifferentialEquation.h"
#include "SteadyStateMathematicalProblem.h"
#include "../InitialConditions/InitialConditions.h"

namespace MathematicalEntities {

    class TransientMathematicalProblem : public SteadyStateMathematicalProblem{
    public:
        TransientMathematicalProblem(shared_ptr<TransientPartialDifferentialEquation> pde,
                                     shared_ptr<DomainBoundaryConditions> bcs,
                                     shared_ptr<InitialConditions> initialConditions,
                                     struct Field_DOFType *degreesOfFreedom
        );
        shared_ptr<TransientPartialDifferentialEquation> transientPDE;
        shared_ptr<InitialConditions> initialConditions;
    };

};

#endif //UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H
