//
// Created by hal9000 on 11/29/22.
//

#include "BoundaryCondition.h"

namespace BoundaryConditions {
    BoundaryCondition::BoundaryCondition(function<double(vector<double>)> *BCFunction) {
        boundaryConditionFunction = BCFunction;
    }
    
    BoundaryCondition::BoundaryCondition(list<tuple<PositioningInSpace::Direction, function<double(vector<double>)>*>> directionalBCFunction) {
        directionalBoundaryConditionFunction = directionalBCFunction;
    }
    
    BoundaryCondition::~BoundaryCondition() {
        delete &boundaryConditionFunction;
        boundaryConditionFunction = nullptr;
        for ( int i = 0; i < directionalBoundaryConditionFunction.size(); i++ ) {
            delete &directionalBoundaryConditionFunction.front();
            directionalBoundaryConditionFunction.pop_front();
        }
    }
    
    
} // BoundaryConditions