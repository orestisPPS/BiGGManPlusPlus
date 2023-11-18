//
// Created by hal9000 on 2/16/23.
//

#include "DomainBoundaryConditions.h"

#include <utility>
#include "../../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace MathematicalEntities {

    DomainBoundaryConditions::DomainBoundaryConditions() {
        _bcAtPosition = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
        _nodalBcAtPosition = nullptr;
    }


    DomainBoundaryConditions::DomainBoundaryConditions(
                shared_ptr<map<Position, shared_ptr<BoundaryCondition>>> bcAtPosition) :
                _bcAtPosition(std::move(bcAtPosition)), _nodalBcAtPosition(nullptr) { }
        
        DomainBoundaryConditions::DomainBoundaryConditions(shared_ptr<map <Position, shared_ptr<map<unsigned int*, shared_ptr<BoundaryCondition>>>>> nodalBcAtPosition) :
                _bcAtPosition(nullptr), _nodalBcAtPosition(std::move(nodalBcAtPosition)) { }
                
        
        shared_ptr<BoundaryCondition> DomainBoundaryConditions::getBoundaryConditionAtPosition(Position boundaryPosition, unsigned* nodeID) {
            if (_bcAtPosition != nullptr) {
                try {
                    return _bcAtPosition->at(boundaryPosition);
                } catch (out_of_range &e) {
                    throw out_of_range("Boundary condition not found at position " + to_string(boundaryPosition));
                }
            } else if (_nodalBcAtPosition != nullptr) {
                try {
                    return _nodalBcAtPosition->at(boundaryPosition)->at(nodeID);
                } catch (out_of_range &e) {
                    throw out_of_range("Boundary condition not found at position " + to_string(boundaryPosition) + " and node ID " + to_string(*nodeID));
                }
            } else {
                throw runtime_error("No boundary conditions found.");
            }
        }

    void DomainBoundaryConditions::setBoundaryCondition(Position boundaryPosition, BoundaryConditionType bcType,
                                                        DOFType dofType, double fixedValue) {
        if (_bcAtPosition == nullptr) {
            _bcAtPosition = make_shared<map<Position, shared_ptr<BoundaryCondition>>>();
        }
        _bcAtPosition->insert(pair<Position, shared_ptr<BoundaryCondition>>(boundaryPosition,
                make_shared<BoundaryCondition>(bcType, make_shared<map<DOFType, double>>( map<DOFType, double>( {{dofType, fixedValue}})))));
    }
    



} // MathematicalEntities