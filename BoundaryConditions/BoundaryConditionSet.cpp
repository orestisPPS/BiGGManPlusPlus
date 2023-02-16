//
// Created by hal9000 on 2/16/23.
//

#include "BoundaryConditionSet.h"

namespace BoundaryConditions {
    BoundaryConditionSet::BoundaryConditionSet() : 
        _boundaryConditions(){
        _boundaryConditions[Dirichlet] = new map<Position, list<BoundaryCondition* >* >();
        _boundaryConditions[Neumann] = new map<Position,list<BoundaryCondition* >* >();
    }
    
    void BoundaryConditionSet::AddDirichletBoundaryConditions(Position boundaryPosition,
                                                              list<BoundaryCondition* >* dirichletBCs){
        _boundaryConditions[Dirichlet]->insert(pair<Position,list<BoundaryCondition* >* >(boundaryPosition,dirichletBCs));
    }
    
    void BoundaryConditionSet::AddNeumannBoundaryConditions(Position boundaryPosition,
                                                            list<BoundaryCondition* >* neumannBCs){
        _boundaryConditions[Neumann]->insert(pair<Position,list<BoundaryCondition* >* >(boundaryPosition,neumannBCs));
    }
    
    
    
} // BoundaryConditions