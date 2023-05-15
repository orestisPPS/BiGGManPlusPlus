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
        
        explicit DomainBoundaryConditions(map <Position, map<unsigned, BoundaryCondition*>>* nodalBcAtPosition);
        
        ~DomainBoundaryConditions();
        
        BoundaryCondition* getBoundaryConditionAtPosition(Position boundaryPosition);
        
        BoundaryCondition* getBoundaryConditionAtPositionAndNode(Position boundaryPosition, unsigned nodeID);
        
        bool varyWithNode() const;
    private:
        bool _varyWithNode;
        
        map <Position, BoundaryCondition*>* _bcAtPosition;

        map<Position, map<unsigned, BoundaryCondition*>>* _nodalBcAtPosition;
    };

} // BoundaryConditions

#endif //UNTITLED_DOMAINBOUNDARYCONDITIONS_H
