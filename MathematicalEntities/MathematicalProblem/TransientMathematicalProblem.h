//
// Created by hal9000 on 12/17/22.
//

#ifndef UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H
#define UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H

#include "../PartialDifferentialEquations/TransientPartialDifferentialEquation.h"
#include "MathematicalProblem.h"
#include "../InitialConditions/InitialConditions.h"

namespace MathematicalEntities {

    class TransientMathematicalProblem : public MathematicalProblem{
    public:
        TransientMathematicalProblem(shared_ptr<TransientPartialDifferentialEquation> pde,
                                     shared_ptr<DomainBoundaryConditions> bcs,
                                     unique_ptr<InitialConditions> initialConditions,
                                     struct Field_DOFType *degreesOfFreedom
        );
        shared_ptr<TransientPartialDifferentialEquation> transientPDE;
        unique_ptr<InitialConditions> _initialConditions;
    };

};

#endif //UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H
