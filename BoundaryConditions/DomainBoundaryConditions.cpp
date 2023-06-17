//
// Created by hal9000 on 2/16/23.
//

#include "DomainBoundaryConditions.h"

#include <utility>
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace BoundaryConditions {
    
        DomainBoundaryConditions::DomainBoundaryConditions(
                shared_ptr<map<Position, shared_ptr<BoundaryCondition>>> bcAtPosition) :
                _bcAtPosition(std::move(bcAtPosition)), _nodalBcAtPosition(nullptr), _varyWithNode(false) { }
        
        DomainBoundaryConditions::DomainBoundaryConditions(shared_ptr<map <Position, map<unsigned, shared_ptr<BoundaryCondition>>>> nodalBcAtPosition) :
                _nodalBcAtPosition(std::move(nodalBcAtPosition)), _bcAtPosition(nullptr), _varyWithNode(true) { }
                
        
        shared_ptr<BoundaryCondition> DomainBoundaryConditions::getBoundaryConditionAtPosition(Position boundaryPosition) {
            return _bcAtPosition->at(boundaryPosition);
        }
        
        shared_ptr<BoundaryCondition> DomainBoundaryConditions::getBoundaryConditionAtPositionAndNode(Position boundaryPosition, unsigned nodeID) {
            return _nodalBcAtPosition->at(boundaryPosition).at(nodeID);
        }
        
        bool DomainBoundaryConditions::varyWithNode() const {
            return _varyWithNode;
        }
        
    
    
} // BoundaryConditions