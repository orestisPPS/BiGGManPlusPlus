//
// Created by hal9000 on 2/18/23.
//

#ifndef UNTITLED_DOFINITIALIZER_H
#define UNTITLED_DOFINITIALIZER_H

#include "../MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../Discretization/Mesh/Mesh.h"

namespace Analysis {

    class DOFInitializer {
    public:
        DOFInitializer(Mesh *mesh,
                       DomainBoundaryConditions *domainBoundaryConditions,
                       list<DegreeOfFreedom*> *degreesOfFreedom);

    list<DegreeOfFreedom*> *freeDegreesOfFreedom;
    list<DegreeOfFreedom*> *boundaryDegreesOfFreedom;
    list<DegreeOfFreedom*> *fixedDegreesOfFreedom;
    list<DegreeOfFreedom*> *degreesOfFreedom;
        
    };

} // DOFInitializer

#endif //UNTITLED_DOFINITIALIZER_H
