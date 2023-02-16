//
// Created by hal9000 on 2/16/23.
//

#ifndef UNTITLED_BOUNDARYCONDITIONSET_H
#define UNTITLED_BOUNDARYCONDITIONSET_H

#include <list>
#include "BoundaryCondition.h"
#include "../PositioningInSpace/DirectionsPositions.h"
using namespace BoundaryConditions;
using namespace PositioningInSpace;
using namespace std;

namespace BoundaryConditions {
    
    enum BoundaryConditionType {
        Dirichlet,
        Neumann
    };
    
    class BoundaryConditionSet {
    public:
        BoundaryConditionSet();
        void AddDirichletBoundaryConditions(Position boundaryPosition, list<BoundaryCondition* >* dirichletBCs);
        void AddNeumannBoundaryConditions(Position boundaryPosition, list<BoundaryCondition* >* neumannBCs);
    //private:
        map <BoundaryConditionType, map<Position,list<BoundaryCondition* >* >* > _boundaryConditions;
    };

} // BoundaryConditions

#endif //UNTITLED_BOUNDARYCONDITIONSET_H
