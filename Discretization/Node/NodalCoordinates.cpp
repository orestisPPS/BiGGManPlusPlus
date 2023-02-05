//
// Created by hal9000 on 2/5/23.
//

#include "NodalCoordinates.h"

namespace Discretization {
    
    NodalCoordinates::NodalCoordinates() : _positionVectors() {
    }
    
    double& NodalCoordinates::operator()(unsigned i) {
        return _positionVectors.at(Natural)(i);
    }   
    
    double& NodalCoordinates::operator()(CoordinateType type, unsigned i) {
        return _positionVectors.at(type)(i);
    }
    
    const double& NodalCoordinates::operator()(unsigned i) const {
        return _positionVectors.at(CoordinateType::Natural)(i);
    }
    
    const double& NodalCoordinates::operator()(CoordinateType type, unsigned i) const {
        return _positionVectors.at(type)(i);
    }
    
    //Adds the input coordinate set type into the node coordinate vector map.
    //Initiated with input vector.
    void NodalCoordinates::addPositionVector(vector<double> positionVector, CoordinateType type) {
        _positionVectors.insert(pair<CoordinateType, CoordinateVector>(type, CoordinateVector(positionVector)));
    }
    
    //Adds a Natural coordinate set the node coordinate vector map.
    //Initiated with input vector.
    void NodalCoordinates::addPositionVector(vector<double> positionVector) {
        _positionVectors.insert(pair<CoordinateType, CoordinateVector>(CoordinateType::Natural, CoordinateVector(positionVector)));
    }
    
    //Adds a coordinate set the node coordinate vector map.
    //The coordinates can be natural, parametric or template.
    //Initiated with NaN values.
    void NodalCoordinates::setPositionVector(CoordinateType type) {
        _positionVectors.insert(pair<CoordinateType, CoordinateVector>(type, CoordinateVector()));
    }
    
    //Replaces the coordinate set of the input type with the input coordinate vector.
    //The coordinates can be natural, parametric or template.
    void NodalCoordinates::changePositionVector(vector<double> positionVector, CoordinateType type) {
        _positionVectors[type] = CoordinateVector(positionVector);
    }
    
    //Replaces the Natural Coordinate set of the input type with the input coordinate vector.
    //The coordinates can be natural, parametric or template.
    void NodalCoordinates::changePositionVector(vector<double> positionVector) {
        _positionVectors[CoordinateType::Natural] = CoordinateVector(positionVector);
    }
    
    //Removes the input coordinate set from the node coordinate vector map.
    void NodalCoordinates::removePositionVector(CoordinateType type) {
        _positionVectors.erase(type);
    }
    
    //Returns the natural position
} // Discretization