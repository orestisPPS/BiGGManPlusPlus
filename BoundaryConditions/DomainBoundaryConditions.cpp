//
// Created by hal9000 on 2/16/23.
//

#include "DomainBoundaryConditions.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace BoundaryConditions {
    DomainBoundaryConditions::DomainBoundaryConditions(SpaceEntityType spaceType) : 
    _boundaryConditions({pair<BoundaryConditionType, map<Position, list<DomainBoundaryConditions *> *> *>
                              (Dirichlet, createBoundaryConditionsMap(spaceType)),
                            pair<BoundaryConditionType, map<Position, list<DomainBoundaryConditions *> *> *>
                                 (Neumann, createBoundaryConditionsMap(spaceType))}) {
    }
    
    map<Position, list<DomainBoundaryConditions *> *> *DomainBoundaryConditions::createBoundaryConditionsMap(SpaceEntityType &spaceType) {
        auto* boundaryConditionsMap = new map<Position, list<DomainBoundaryConditions* >* >();
        switch (spaceType) {
            case PositioningInSpace::Axis:
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions *> *>(Left, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Right, new list<DomainBoundaryConditions* >()));
                break;
            case PositioningInSpace::Plane:
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Left, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Right, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Bottom, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Top, new list<DomainBoundaryConditions* >()));
                break;
            case PositioningInSpace::Volume:
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Left, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Right, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Bottom, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Top, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Front, new list<DomainBoundaryConditions* >()));
                boundaryConditionsMap->insert(pair<Position, list<DomainBoundaryConditions* > *>(Back, new list<DomainBoundaryConditions* >()));
                break;
            default:
                throw invalid_argument("Invalid space type");
            case NullSpace:
                break;
        }
        return boundaryConditionsMap;
    }
    
    void DomainBoundaryConditions::AddDirichletBoundaryConditions(Position boundaryPosition,
                                                                  list<DomainBoundaryConditions* >* dirichletBCs){
        _boundaryConditions[Dirichlet]->insert(
                pair<Position, list<DomainBoundaryConditions* > *>(boundaryPosition, dirichletBCs));
    }
    
    void DomainBoundaryConditions::AddNeumannBoundaryConditions(Position boundaryPosition,
                                                                list<DomainBoundaryConditions* >* neumannBCs){
        _boundaryConditions[Neumann]->insert(
                pair<Position, list<DomainBoundaryConditions* > *>(boundaryPosition, neumannBCs));
    }
    
    
    
} // BoundaryConditions