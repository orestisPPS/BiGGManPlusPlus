//
// Created by hal9000 on 2/16/23.
//

#ifndef UNTITLED_DOMAINBOUNDARYCONDITIONS_H
#define UNTITLED_DOMAINBOUNDARYCONDITIONS_H

#include <list>
#include<exception>
#include <stdexcept>
#include "BoundaryCondition.h"
#include "../PositioningInSpace/DirectionsPositions.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

using namespace PositioningInSpace;
using namespace DegreesOfFreedom;
using namespace std;

namespace BoundaryConditions {
    
    class DomainBoundaryConditions {
    public:
        explicit DomainBoundaryConditions(map <Position, BoundaryCondition*>* bcAtPosition);
        BoundaryCondition* getBoundaryConditionAtPosition(Position boundaryPosition);
    private:
        map <Position, BoundaryCondition*>* bcAtPosition;
    };

} // BoundaryConditions

#endif //UNTITLED_DOMAINBOUNDARYCONDITIONS_H
