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
        DOFInitializer(Mesh *mesh,DomainBoundaryConditions *domainBoundaryConditions,struct Field_DOFType *degreesOfFreedom);
        list<DegreeOfFreedom*> *freeDegreesOfFreedom;
        list<DegreeOfFreedom*> *boundedDegreesOfFreedom;
        list<DegreeOfFreedom*> *fluxDegreesOfFreedom;
        
    private:
        void addDOFToInternalNodes(Mesh *mesh, list<DOFType*>* degreesOfFreedom );
        void addDOFToBoundaryNodes(Mesh *mesh, list<DOFType*>* degreesOfFreedom );
    };

} // DOFInitializer

#endif //UNTITLED_DOFINITIALIZER_H
