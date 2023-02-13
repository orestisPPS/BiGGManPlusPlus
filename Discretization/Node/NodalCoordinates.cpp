//
// Created by hal9000 on 2/5/23.
//

#include "NodalCoordinates.h"

#include <utility>

namespace Discretization {
    
    NodalCoordinates::NodalCoordinates() : _positionVectors() {
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
        _positionVectors.insert(pair<CoordinateType, CoordinateVector>(type, CoordinateVector(std::move(positionVector))));
    }
    
    //Adds a Natural coordinate set the node coordinate vector map.
    //Initiated with input vector.
    void NodalCoordinates::addPositionVector(vector<double> positionVector) {
        _positionVectors.insert(pair<CoordinateType, CoordinateVector>(CoordinateType::Natural, CoordinateVector(std::move(positionVector))));
    }
    
    void NodalCoordinates::addPositionVector(CoordinateType type) {
        _positionVectors.insert(pair<CoordinateType, CoordinateVector>(type, CoordinateVector()));
    }
    

    //Replaces the coordinate set of the input type with the input coordinate vector.
    //The coordinates can be natural, parametric or template.
    void NodalCoordinates::setPositionVector(vector<double> positionVector, CoordinateType type) {
        _positionVectors[type] = CoordinateVector(std::move(positionVector));
    }
    
    //Replaces the Natural Coordinate set of the input type with the input coordinate vector.
    //The coordinates can be natural, parametric or template.
    void NodalCoordinates::setPositionVector(vector<double> positionVector) {
        _positionVectors[CoordinateType::Natural] = CoordinateVector(std::move(positionVector));
    }
    
    //Removes the input coordinate set from the node coordinate vector map.
    void NodalCoordinates::removePositionVector(CoordinateType type) {
        _positionVectors.erase(type);
    }
    
    //Returns the natural position vector of the Node 
    const vector<double>& NodalCoordinates::positionVector() {
        return *( _positionVectors.at(CoordinateType::Natural).getCoordinateVector());
    }
    
    //Returns the input position vector of the Node 
    const vector<double>& NodalCoordinates::positionVector(CoordinateType type) {
        return *( _positionVectors.at(type).getCoordinateVector());
    }
    
    //Size of the natural position vector
    unsigned NodalCoordinates::size() {
    return  _positionVectors.at(CoordinateType::Natural).dimensions();
    }
    
    //Size of the input position vector
    unsigned NodalCoordinates::size(CoordinateType type) {
        return  _positionVectors.at(type).dimensions();
    }
        
    //Returns the natural position
} // Discretization