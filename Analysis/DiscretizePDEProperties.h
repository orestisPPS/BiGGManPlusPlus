//
// Created by hal9000 on 2/16/23.
//

#ifndef UNTITLED_DISCRETIZEPDEPROPERTIES_H
#define UNTITLED_DISCRETIZEPDEPROPERTIES_H

#include "../MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../Discretization/Mesh/Mesh.h"

namespace Analysis {

    class DiscretizePDEProperties {
    public:
        DiscretizePDEProperties();
    private:
        Mesh *mesh;
        map<Position,list<BoundaryConditions::BoundaryCondition*>*> *boundaryConditions;
        //list<DegreeOfFreedom*> *degreesOfFreedom;
    };

} // DOFInitializer

#endif //UNTITLED_DISCRETIZEPDEPROPERTIES_H
