//
// Created by hal9000 on 11/28/22.
//

#include <iostream>
#include <list>
#include "Node.h"
#include "../Id/DiscreteEntityId.h"
using namespace Discretization;

namespace Discretization
{
    Node::Node(PhysicalSpaceEntity spaceEntity) {
        space = spaceEntity;
    }
    
    void Node::setPositionVector(vector<double> positionVector, CoordinateType type) {
        auto coordinateVector =
                _position.insert(pair<CoordinateType, CoordinateVector>
                                         (type, CoordinateVector(std::move(positionVector), space.type())));
    }

    void Node::removePositionVector(CoordinateType type) {
        _position.erase(type);
    }
    
    vector<double> Node::positionVector() {
        return _position.at(CoordinateType::Natural).getCoordinateVectorIn3D(space.type());
    }
    
    vector<double> Node::positionVector(CoordinateType type) {
        return _position.at(type).getCoordinateVectorIn3D(space.type());
    }
    
    vector<double> Node::positionVector(PhysicalSpaceEntities physicalSpace) {
        return _position.at(CoordinateType::Natural).getCoordinateVectorInEntity(space.type(), physicalSpace);
    }
    
    vector<double> Node::positionVector(CoordinateType type, PhysicalSpaceEntities physicalSpace) {
            return _position.at(type).getCoordinateVectorInEntity(space.type(), physicalSpace);
    }
    
    unsigned Node::positionVectorDimensions() {
        return _position.at(CoordinateType::Natural).dimensions();
    }
    
    unsigned Node::positionVectorDimensions(CoordinateType type) {
        return _position.at(type).dimensions();
    }
}// Discretization
    

