//
// Created by hal9000 on 2/16/23.
//
#include <stdexcept>
#include "BoundaryCondition.h"

namespace BoundaryConditions {
    BoundaryCondition::BoundaryCondition(function<double(vector<double>*)> BCFunction) :
            _boundaryConditionFunction(std::move(BCFunction)){
    }

    BoundaryCondition::BoundaryCondition(map<Direction, function<double(vector<double>*)>> directionalBCFunction) :
            _directionalBoundaryConditionFunction(std::move(directionalBCFunction)) {
        _checkDirectionalBoundaryConditionFunction();
    }

    double BoundaryCondition::scalarValueAt(vector<double> *coordinates) {
        if (coordinates == nullptr)
            throw std::invalid_argument("Coordinates cannot be null.");
        else
            return _boundaryConditionFunction(coordinates);
    }

    double BoundaryCondition::vectorValueAtDirection(Direction direction, vector<double> *coordinates) {
        return _directionalBoundaryConditionFunction.at(direction)(coordinates);
    }
    
    vector<double> BoundaryCondition::vectorValueAt(vector<double> *coordinates) {
        vector<double> vectorValue;
        vectorValue.at(0) = _directionalBoundaryConditionFunction.at(Direction::One)(coordinates);
        if (_directionalBoundaryConditionFunction.find(Direction::Two) != _directionalBoundaryConditionFunction.end())
            vectorValue.at(1) = _directionalBoundaryConditionFunction.at(Direction::Two)(coordinates);
        if (_directionalBoundaryConditionFunction.find(Direction::Three) != _directionalBoundaryConditionFunction.end())
            vectorValue.at(2) = _directionalBoundaryConditionFunction.at(Direction::Three)(coordinates);
        return vectorValue;
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