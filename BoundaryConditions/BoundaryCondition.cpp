//
// Created by hal9000 on 11/29/22.
//

#include "BoundaryCondition.h"
#include <limits>
#include <iostream>
#include <utility>

namespace BoundaryConditions {
    
    BoundaryCondition::BoundaryCondition(function<double(vector<double>*)> BCFunction) : 
    _boundaryConditionFunction(std::move(BCFunction)){
    }
    
    BoundaryCondition::BoundaryCondition(map<Direction, function<double(vector<double>*)>> directionalBCFunction) :
    _directionalBoundaryConditionFunction(std::move(directionalBCFunction)) {
    }
    
    double BoundaryCondition::valueAt(vector<double> *x) {
        if (x == nullptr)
            throw std::invalid_argument("Input is a null pointer.");
        
        else
            return _boundaryConditionFunction(x);
    }
    
    double BoundaryCondition::valueAt(Direction direction, vector<double> *x) {
        return _directionalBoundaryConditionFunction.at(direction)(x);
    }   
    
    void BoundaryCondition::_checkDirectionalBoundaryConditionFunction() {
        if (_directionalBoundaryConditionFunction.empty()) {
            throw std::invalid_argument("At least one direction must be specified for a directional boundary condition.");
        }
        if (_directionalBoundaryConditionFunction.size() > 3) {
            throw std::invalid_argument("A boundary condition can only be specified in Direction:: One/Two/Three.");
        }
        //Check for time boundary condition
        if (_directionalBoundaryConditionFunction.find(Direction::Time) != _directionalBoundaryConditionFunction.end()) {
            throw std::invalid_argument("A boundary condition cannot be specified in Direction::Time. It is an initial condition.");
        }
    }
} // BoundaryConditions