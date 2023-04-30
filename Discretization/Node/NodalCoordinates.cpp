//
// Created by hal9000 on 2/5/23.
//

#include "NodalCoordinates.h"

#include <utility>

namespace Discretization {
    
    NodalCoordinates::NodalCoordinates() : _positionVectors(new map<CoordinateType, vector<double>*>()) {
    }
    
    NodalCoordinates::~NodalCoordinates() {
        for (auto &positionVector : *_positionVectors) {
            delete positionVector.second;
        }
        _positionVectors->clear();
        delete _positionVectors;
    }
    
    const double& NodalCoordinates::operator()(unsigned i) const {
        if (_positionVectors->at(Natural)->size() <= i)
            throw runtime_error("Node coordinate not found!");
        return _positionVectors->at(Natural)->at(i);
    }
    
    const double& NodalCoordinates::operator()(CoordinateType type, unsigned i) const {
        if (_positionVectors->at(type)->size() <= i)
            throw runtime_error("Node coordinate not found!");
        return _positionVectors->at(type)->at(i);
    }
    
    //Adds the input coordinate set type into the node coordinate vector map.
    //Initiated with input vector.
    void NodalCoordinates::addPositionVector(vector<double>* positionVector, CoordinateType type) {
        _positionVectors->insert(pair<CoordinateType, vector<double>*>(type, positionVector));
    }
    
    //Adds a Natural coordinate set the node coordinate vector map.
    //Initiated with input vector.
    void NodalCoordinates::addPositionVector(vector<double>* positionVector) {
        if (positionVector->empty() && positionVector->size()<= 3)
            _positionVectors->insert(pair<CoordinateType, vector<double>*>(Natural, positionVector));
        else
            _positionVectors->insert(pair<CoordinateType, vector<double>*>(Natural, positionVector));    }
    
    void NodalCoordinates::addPositionVector(CoordinateType type) {
        _positionVectors->insert(pair<CoordinateType, vector<double>*>(type, new vector<double>));
    }
    

    //Replaces the coordinate set of the input type with the input coordinate vector.
    //The coordinates can be natural, parametric or template.
    void NodalCoordinates::setPositionVector(vector<double>*positionVector, CoordinateType type) {
        if (positionVector->empty() && positionVector->size()<= 3)
            _positionVectors->insert(pair<CoordinateType, vector<double>*>(type, positionVector));
        else
            _positionVectors->insert(pair<CoordinateType, vector<double>*>(type, positionVector));
    }
    
    //Replaces the Natural Coordinate set of the input type with the input coordinate vector.
    //The coordinates can be natural, parametric or template.
    void NodalCoordinates::setPositionVector(vector<double>* positionVector) {
        if (positionVector->empty() && positionVector->size()<= 3)
            _positionVectors->insert(pair<CoordinateType, vector<double>*>(Natural, positionVector));
        else
            _positionVectors->insert(pair<CoordinateType, vector<double>*>(Natural, positionVector));
    }
    
    //Removes the input coordinate set from the node coordinate vector map.
    void NodalCoordinates::removePositionVector(CoordinateType type) {
        _positionVectors->at(type)->clear();
        delete _positionVectors->at(type);
        _positionVectors->erase(type);
    }
    
    //Returns the natural position vector of the Node 
    const vector<double>& NodalCoordinates::positionVector() {
        return *( _positionVectors->at(Natural));
    }
    
    vector<double>* NodalCoordinates::positionVectorPtr() {
        return _positionVectors->at(Natural);
    }
        
    //Returns the input position vector of the Node 
    const vector<double>& NodalCoordinates::positionVector(CoordinateType type) {
        return *( _positionVectors->at(type));
    }
    
    //Returns a pointer to the input position vector of the Node
    vector<double>* NodalCoordinates::positionVectorPtr(CoordinateType type) {
        return _positionVectors->at(type);
    }
    
} // Discretization