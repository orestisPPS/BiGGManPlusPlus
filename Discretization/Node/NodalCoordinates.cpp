//
// Created by hal9000 on 2/5/23.
//



#include "NodalCoordinates.h"

#include <utility>

namespace Discretization {
    
    NodalCoordinates::NodalCoordinates() :
    _positionVectors(make_unique<map<CoordinateType, shared_ptr<NumericalVector<double>>>>()) { }
    
    NodalCoordinates::~NodalCoordinates() {
        for (auto &positionVector : *_positionVectors) {
            positionVector.second->clear();
        }
        _positionVectors->clear();
    }
    
    NodalCoordinates& NodalCoordinates::operator=(const NodalCoordinates& other){
        if (this != &other) {  // Protect against self-assignment
            // Deep copy the _positionVectors map
            _positionVectors = make_unique<map<CoordinateType, shared_ptr<NumericalVector<double>>>>();

            for (const auto& tuple : *other._positionVectors) {
                // Make a new NumericalVector using the data from the other object
                shared_ptr<NumericalVector<double>> newVector = make_shared<NumericalVector<double>>(*tuple.second);
                _positionVectors->insert(pair<CoordinateType, shared_ptr<NumericalVector<double>>>(tuple.first, newVector));
            }
        }
        return *this;
    }
    
    const double& NodalCoordinates::operator()(unsigned i, CoordinateType type) const {
        if (_positionVectors->at(type)->size() <= i)
            throw runtime_error("Node coordinate not found!");
        return _positionVectors->at(type)->at(i);
    }
    
    const shared_ptr<NumericalVector<double>>& NodalCoordinates::getPositionVector(CoordinateType type) {
        return _positionVectors->at(type);
    }
    
    NumericalVector<double> NodalCoordinates::getPositionVector3D(CoordinateType type) {
        NumericalVector<double> coords = *_positionVectors->at(type);
        switch (coords.size()) {
            case 1:
                return NumericalVector<double>({coords[0], 0.0, 0.0});
            case 2:
                return NumericalVector<double>({coords[0], coords[1], 0.0});
            case 3:
                return coords;
            default:
                throw runtime_error("Node coordinate not found!");

        }
    }
            
    void NodalCoordinates::setPositionVector(shared_ptr<NumericalVector<double>>positionVector, CoordinateType type) {
        if (_positionVectors->find(type) != _positionVectors->end())
            _positionVectors->at(type) = std::move(positionVector);
        else
            _positionVectors->insert(pair<CoordinateType, shared_ptr<NumericalVector<double>>>(type, std::move(positionVector)));
    }

    void NodalCoordinates::removePositionVector(CoordinateType type) {
        if (_positionVectors->find(type) != _positionVectors->end())
            _positionVectors->erase(type);
    }
    
    
} // Discretization