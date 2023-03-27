//
// Created by hal9000 on 2/16/23.
//

#include "DomainBoundaryConditions.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace BoundaryConditions {
    
        DomainBoundaryConditions::DomainBoundaryConditions(map <Position, BoundaryCondition*>* bcAtPosition) :
                bcAtPosition(bcAtPosition) {
            DomainBoundaryConditions::bcAtPosition = bcAtPosition;
        }
    
        BoundaryCondition* DomainBoundaryConditions::getBoundaryConditionAtPosition(Position boundaryPosition) {
            return bcAtPosition->at(boundaryPosition);
        }
        
    
    
} // BoundaryConditions