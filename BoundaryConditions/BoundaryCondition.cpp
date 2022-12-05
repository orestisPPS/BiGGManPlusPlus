//
// Created by hal9000 on 11/29/22.
//

#include "BoundaryCondition.h"
#include <limits>
#include <iostream>

namespace BoundaryConditions {
    BoundaryCondition::BoundaryCondition(function<double(vector<double>)> *BCFunction) {
        _boundaryConditionFunction = BCFunction;
    }
    
    BoundaryCondition::BoundaryCondition(list<tuple<PositioningInSpace::Direction, function<double(vector<double>)>*>> *directionalBCFunction) {
        _directionalBoundaryConditionFunction = directionalBCFunction;
    }
    
    BoundaryCondition::~BoundaryCondition() {
        if (_boundaryConditionFunction != nullptr) {
            delete _boundaryConditionFunction;
            _boundaryConditionFunction = nullptr;
        }
        if (_directionalBoundaryConditionFunction != nullptr) {
            for (int i = 0; i < _directionalBoundaryConditionFunction->size(); ++i) {
                delete &get<1>(_directionalBoundaryConditionFunction->front());
                _directionalBoundaryConditionFunction->pop_front();
            }
            delete _directionalBoundaryConditionFunction;
        }
    }
    
    double BoundaryCondition::valueAt(vector<double> &x) {
        if (_boundaryConditionFunction != nullptr) {
            return (*_boundaryConditionFunction)(x);
        } else {
            std::cout << "Error: No boundary condition function defined." << std::endl;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double BoundaryCondition::valueAt(Direction direction, vector<double> &x) {
        if (_directionalBoundaryConditionFunction != nullptr) {
            for (auto &directionalBCFunction : *_directionalBoundaryConditionFunction) {
                if (get<0>(directionalBCFunction) == direction) {
                    return (*get<1>(directionalBCFunction))(x);
                }
            }
        }
        else {
            std::cout << "Error: No directional boundary condition function defined." << std::endl;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    
} // BoundaryConditions