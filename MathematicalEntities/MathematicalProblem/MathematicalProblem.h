//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_MATHEMATICALPROBLEM_H
#define UNTITLED_MATHEMATICALPROBLEM_H

#include "../../MathematicalEntities/PartialDifferentialEquations/PartialDifferentialEquation.h"
#include "../../MathematicalEntities/BoundaryConditions/DomainBoundaryConditions.h"
#include "../../DegreesOfFreedom/DegreeOfFreedomTypes.h"
using namespace DegreesOfFreedom;
using namespace MathematicalEntities;

namespace MathematicalEntities {

    class MathematicalProblem {
        public:
        MathematicalProblem(shared_ptr<PartialDifferentialEquation> pde,
                            shared_ptr<DomainBoundaryConditions> bcs,
                            Field_DOFType* degreesOfFreedom);
        
        shared_ptr<PartialDifferentialEquation> pde;
        shared_ptr<DomainBoundaryConditions> boundaryConditions;
        Field_DOFType *degreesOfFreedom;
    };

} // MathematicalEntities

#endif //UNTITLED_MATHEMATICALPROBLEM_H
