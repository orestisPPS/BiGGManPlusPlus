//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_MATHEMATICALPROBLEM_H
#define UNTITLED_MATHEMATICALPROBLEM_H

#include "../PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../BoundaryConditions/DomainBoundaryConditions.h"
#include "../DegreesOfFreedom/DegreeOfFreedomTypes.h"

namespace MathematicalProblem {

    class MathematicalProblem {
        public:
        MathematicalProblem(PartialDifferentialEquation* pde,
                            DomainBoundaryConditions* bcs,
                            struct Field_DOFType* degreesOfFreedom);
        ~MathematicalProblem();

        PartialDifferentialEquation* pde;
        DomainBoundaryConditions* boundaryConditions;
        Field_DOFType *degreesOfFreedom;
    };

} // MathematicalProblem

#endif //UNTITLED_MATHEMATICALPROBLEM_H
