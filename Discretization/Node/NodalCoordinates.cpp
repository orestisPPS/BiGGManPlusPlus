//
// Created by hal9000 on 2/5/23.
//

#include "NodalCoordinates.h"

#include <utility>

namespace Discretization {
    
    NodalCoordinates::NodalCoordinates() :
    _positionVectors(make_shared<map<CoordinateType, shared_ptr<vector<double>>>>()) { }
    
    NodalCoordinates::~NodalCoordinates() {
        for (auto &positionVector : *_positionVectors) {
            positionVector.second->clear();
        }
        _positionVectors->clear();
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
    

    void NodalCoordinates::addPositionVector(shared_ptr<vector<double>> positionVector, CoordinateType type) {
        _positionVectors->insert(pair<CoordinateType, shared_ptr<vector<double>>>(type, std::move(positionVector)));
    }
    
    //Adds a Natural coordinate set the node coordinate vector map.
    //Initiated with input vector.
    void NodalCoordinates::addPositionVector(shared_ptr<vector<double>> positionVector) {
        if (positionVector->empty() && positionVector->size()<= 3)
            _positionVectors->insert(pair<CoordinateType, shared_ptr<vector<double>>>(Natural, std::move(positionVector)));
        else
            _positionVectors->insert(pair<CoordinateType, shared_ptr<vector<double>>>(Natural, std::move(positionVector)));    }
    
    void NodalCoordinates::addPositionVector(CoordinateType type) {
        _positionVectors->insert(pair<CoordinateType, shared_ptr<vector<double>>>(type, new vector<double>));
    }
    

    void NodalCoordinates::setPositionVector(shared_ptr<vector<double>>positionVector, CoordinateType type) {
        if (positionVector->empty() && positionVector->size()<= 3)
            _positionVectors->insert(pair<CoordinateType, shared_ptr<vector<double>>>(type, std::move(positionVector)));
        else
            _positionVectors->insert(pair<CoordinateType, shared_ptr<vector<double>>>(type, std::move(positionVector)));
    }
    
    void NodalCoordinates::setPositionVector(shared_ptr<vector<double>> positionVector) {
        _positionVectors->insert(pair<CoordinateType, shared_ptr<vector<double>>>(Natural, std::move(positionVector)));
    }
    
    void NodalCoordinates::removePositionVector(CoordinateType type) {
        _positionVectors->at(type)->clear();
        _positionVectors->erase(type);
    }
    
    const vector<double>& NodalCoordinates::positionVector() {
        return *( _positionVectors->at(Natural));
    }
     
    const vector<double>& NodalCoordinates::positionVector(CoordinateType type) {
        return *( _positionVectors->at(type));
    }
    
    const shared_ptr<vector<double>>& NodalCoordinates::positionVectorPtr(CoordinateType type) {
        return _positionVectors->at(type);
    }
    
    vector<double> NodalCoordinates::positionVector3D() {

        auto coords = positionVector();
        switch (coords.size()) {
            case 1:
                return {coords[0], 0.0, 0.0};
            case 2:
                return {coords[0], coords[1], 0.0};
            case 3:
                return {coords[0], coords[1], coords[2]};
            default:
                throw runtime_error("Node coordinate not found!");
            
        }
    }
    
    vector<double> NodalCoordinates::positionVector3D(CoordinateType type) {
        auto coords = positionVector(type);
        switch (coords.size()) {
            case 1:
                return {coords[0], 0.0, 0.0};
            case 2:
                return {coords[0], coords[1], 0.0};
            case 3:
                return {coords[0], coords[1], coords[2]};
            default:
                throw runtime_error("Node coordinate not found!");
            
        }
    }
    
    shared_ptr<vector<double>> NodalCoordinates::positionVector3DPtr() {
        auto coords = positionVector();
        switch (coords.size()) {
            case 1:
                return shared_ptr<vector<double>>(new vector<double>{coords[0], 0.0, 0.0});
            case 2:
                return shared_ptr<vector<double>>(new vector<double>{coords[0], coords[1], 0.0});
            case 3:
                return shared_ptr<vector<double>>(new vector<double>{coords[0], coords[1], coords[2]});
            default:
                throw runtime_error("Node coordinate not found!");
            
        }
    }
    
    shared_ptr<vector<double>> NodalCoordinates::positionVector3DPtr(CoordinateType type) {
        auto coords = positionVector(type);
        switch (coords.size()) {
            case 1:
                return shared_ptr<vector<double>>(new vector<double>{coords[0], 0.0, 0.0});
            case 2:
                return shared_ptr<vector<double>>(new vector<double>{coords[0], coords[1], 0.0});
            case 3:
                return shared_ptr<vector<double>>(new vector<double>{coords[0], coords[1], coords[2]});
            default:
                throw runtime_error("Node coordinate not found!");
            
        }
    }
    
    
    
    
} // Discretization