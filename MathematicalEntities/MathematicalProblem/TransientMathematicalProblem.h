//
// Created by hal9000 on 12/17/22.
//

#ifndef UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H
#define UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H

#include "SteadyStateMathematicalProblem.h"
#include "../InitialConditions/InitialConditions.h"

namespace MathematicalEntities {

    class TransientMathematicalProblem : public MathematicalProblem{
    public:
        TransientMathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                                     shared_ptr<DomainBoundaryConditions> bcs,
                                     unique_ptr<InitialConditions> initialConditions,
                                     struct Field_DOFType *degreesOfFreedom
        );
        
        unique_ptr<InitialConditions> _initialConditions;
    };

};

#endif //UNUNTITLED_TRANSIENTMATHEMATICALPROBLEM_H
