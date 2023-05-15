//
// Created by hal9000 on 2/16/23.
//

#include "DomainBoundaryConditions.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace BoundaryConditions {
    
        DomainBoundaryConditions::DomainBoundaryConditions(map <Position, BoundaryCondition*>* bcAtPosition) :
                _bcAtPosition(bcAtPosition), _nodalBcAtPosition(nullptr), _varyWithNode(false) { }
        
        DomainBoundaryConditions::DomainBoundaryConditions(map <Position, map<unsigned, BoundaryCondition*>>* nodalBcAtPosition) :
                _nodalBcAtPosition(nodalBcAtPosition), _bcAtPosition(nullptr), _varyWithNode(true) { }
                
        DomainBoundaryConditions::~DomainBoundaryConditions() {
            if (_bcAtPosition != nullptr) {
                for (auto &bc : *_bcAtPosition) {
                    delete bc.second;
                }
                delete _bcAtPosition;
            }
            if (_nodalBcAtPosition != nullptr) {
                for (auto &bc : *_nodalBcAtPosition) {
                    for (auto &nodeBC : bc.second) {
                        delete nodeBC.second;
                    }
                }
                delete _nodalBcAtPosition;
            }
        }
        
        BoundaryCondition* DomainBoundaryConditions::getBoundaryConditionAtPosition(Position boundaryPosition) {
            return _bcAtPosition->at(boundaryPosition);
        }
        
        BoundaryCondition* DomainBoundaryConditions::getBoundaryConditionAtPositionAndNode(Position boundaryPosition, unsigned nodeID) {
            return _nodalBcAtPosition->at(boundaryPosition).at(nodeID);
        }
        
        bool DomainBoundaryConditions::varyWithNode() const {
            return _varyWithNode;
        }
        
    
    
} // BoundaryConditions