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

using namespace BoundaryConditions;
using namespace PositioningInSpace;
using namespace std;

namespace BoundaryConditions {
    
    enum BoundaryConditionType {
        Dirichlet,
        Neumann
    };
    
    class DomainBoundaryConditions {
    public:
        explicit DomainBoundaryConditions(SpaceEntityType spaceType);
        void AddDirichletBoundaryConditions(Position boundaryPosition, list<BoundaryCondition* >* dirichletBCs);
        void AddNeumannBoundaryConditions(Position boundaryPosition, list<BoundaryCondition* >* neumannBCs);
        list<BoundaryCondition*>* GetBoundaryConditions(Position boundaryPosition, BoundaryConditionType boundaryConditionType);
        map <BoundaryConditionType, map<Position,list<BoundaryCondition* >* >* > boundaryConditions;

    private:
        static map<Position,list<BoundaryCondition* >* >* createBoundaryConditionsMap(SpaceEntityType& spaceType);
    };

} // BoundaryConditions

#endif //UNTITLED_DOMAINBOUNDARYCONDITIONS_H
