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
        explicit DomainBoundaryConditions( shared_ptr<map<Position, shared_ptr<BoundaryCondition>>> bcAtPosition);
        
        explicit DomainBoundaryConditions(shared_ptr<map <Position, map<unsigned, shared_ptr<BoundaryCondition>>>> nodalBcAtPosition);
        
        shared_ptr<BoundaryCondition> getBoundaryConditionAtPosition(Position boundaryPosition);
        
        shared_ptr<BoundaryCondition> getBoundaryConditionAtPositionAndNode(Position boundaryPosition, unsigned nodeID);
        
        bool varyWithNode() const;
    private:
        bool _varyWithNode;

        shared_ptr<map<Position, map<unsigned int, shared_ptr<BoundaryCondition>>>> _nodalBcAtPosition;

        shared_ptr<map<Position, shared_ptr<BoundaryCondition>>> _bcAtPosition;
    };

} // BoundaryConditions

#endif //UNTITLED_DOMAINBOUNDARYCONDITIONS_H
