//
// Created by hal9000 on 2/16/23.
//

#include "DomainBoundaryConditions.h"
#include "../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace BoundaryConditions {
    DomainBoundaryConditions::DomainBoundaryConditions(SpaceEntityType spaceType) :
            boundaryConditions({pair<BoundaryConditionType, map<Position, list<BoundaryCondition *> *> *>
                              (Dirichlet, createBoundaryConditionsMap(spaceType)),
                                pair<BoundaryConditionType, map<Position, list<BoundaryCondition *> *> *>
                                 (Neumann, createBoundaryConditionsMap(spaceType))}) {
    }

    map <Position, tuple<BoundaryConditionType, map<DOFType*, BoundaryCondition*>*>*>*DomainBoundaryConditions::createBoundaryConditionsMap(SpaceEntityType &spaceType) {
        auto* boundaryConditionsMap = new map <Position, tuple<BoundaryConditionType, map<DOFType*, BoundaryCondition*>*>*>*;
        switch (spaceType) {
            case PositioningInSpace::Axis:
                
                boundaryConditionsMap->insert(pair<Position, tuple<BoundaryConditionType, map<DOFType*, BoundaryCondition*>*>*>
                        (Left, new tuple<BoundaryConditionType, map<DOFType*, BoundaryCondition*>*>(Dirichlet, new map<DOFType*, BoundaryCondition*>())));
                
                boundaryConditionsMap->insert(pair<Position,list<BoundaryCondition *> *>(Left, new list<BoundaryCondition* >()));
                boundaryConditionsMap->insert(pair<Position,list<BoundaryCondition* > *>(Right, new list<BoundaryCondition* >()));
                break; 
            case PositioningInSpace::Plane:
                boundaryConditionsMap->insert(pair<Position,list<BoundaryCondition* > *>(Left, new list<BoundaryCondition* >()));
                boundaryConditionsMap->insert(pair<Position,list<BoundaryCondition* > *>(Right, new list<BoundaryCondition* >()));
                boundaryConditionsMap->insert(pair<Position,list<BoundaryCondition* > *>(Bottom, new list<BoundaryCondition* >()));
                boundaryConditionsMap->insert(pair<Position,list<BoundaryCondition* > *>(Top, new list<BoundaryCondition* >()));
                break;
            case PositioningInSpace::Volume:
                
                break;
            default:
                throw invalid_argument("Invalid space type");
            case NullSpace:
                break;
        }
        return boundaryConditionsMap;
    }
    
    void DomainBoundaryConditions::AddDirichletBoundaryConditions(Position boundaryPosition,
                                                                  list<BoundaryCondition* >* dirichletBCs){
        boundaryConditions[Dirichlet]->at(boundaryPosition) = (dirichletBCs);
        boundaryConditions[Neumann]->erase(boundaryPosition);
    }
    
    void DomainBoundaryConditions::AddNeumannBoundaryConditions(Position boundaryPosition,
                                                                list<BoundaryCondition* >* neumannBCs){
        boundaryConditions[Neumann]->at(boundaryPosition) = (neumannBCs);
        boundaryConditions[Dirichlet]->erase(boundaryPosition);
    }
    
    list<BoundaryCondition* >* DomainBoundaryConditions::GetBoundaryConditions(Position boundaryPosition,
                                                                                      BoundaryConditionType boundaryConditionType){
        if (boundaryConditions[boundaryConditionType]->find(boundaryPosition) != boundaryConditions[boundaryConditionType]->end()){
            return boundaryConditions[boundaryConditionType]->at(boundaryPosition);
        }
        else{
            return nullptr;
        }
    }
    
    
    
} // BoundaryConditions