//
// Created by hal9000 on 11/28/22.
//

#include <iostream>
#include <list>
#include <utility>
#include "Node.h"
#include "../Id/DiscreteEntityId.h"
using namespace Discretization;

namespace Discretization {
    Node::Node(PhysicalSpaceEntity physicalSpace) : space(std::move(physicalSpace)) {
        id = DiscreteEntityId();
    }

    void Node::setPositionVector(vector<double> positionVector, CoordinateType type) {
        _position.insert(pair<CoordinateType, CoordinateVector>(type, CoordinateVector(std::move(positionVector), space.type())));
    }

    void Node::setPositionVector(CoordinateType type) {
        _position.insert(pair<CoordinateType, CoordinateVector>(type, CoordinateVector(space.type())));
    }

    void Node::setPositionVector(vector<double> positionVector) {
        auto coordinateVector =
                _position.insert(pair<CoordinateType, CoordinateVector>
                                         (Natural, CoordinateVector(std::move(positionVector), space.type())));
    }

    void Node::changePositionVector(vector<double> positionVector, CoordinateType type) {
        _position.at(type).setCoordinateVector(positionVector, space.type());
    }
    
    void Node::changePositionVector(vector<double> positionVector) {
        _position.at(Natural).setCoordinateVector(positionVector, space.type());
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

    vector<double> Node::positionVector(SpaceEntityType physicalSpace) {
        return _position.at(CoordinateType::Natural).getCoordinateVectorInEntity(space.type(), physicalSpace);
    }

    vector<double> Node::positionVector(CoordinateType type, SpaceEntityType physicalSpace) {
        return _position.at(type).getCoordinateVectorInEntity(space.type(), physicalSpace);
    }

    unsigned Node::positionVectorDimensions() {
        return _position.at(CoordinateType::Natural).dimensions();
    }

    unsigned Node::positionVectorDimensions(CoordinateType type) {
        return _position.at(type).dimensions();
    }
}// Discretization
    

