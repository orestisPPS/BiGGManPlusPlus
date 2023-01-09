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
        if (space.type() == PhysicalSpaceEntities::OneTwoThree_volume) {
            
        }
        else {
            _position = CoordinateVector(0);
        } 
    }


} // Discretization
    


/*
Node::Node() {
    id = new DiscreteEntityId();
    _coordinates = new std::list<Coordinate*>();
    _degreesOfFreedom = new std::list<DegreeOfFreedom*>();
}

Node::~Node() {
    delete id;
    id = nullptr;
    delete _coordinates;
    _coordinates = nullptr;
    delete _degreesOfFreedom;
    _degreesOfFreedom = nullptr;
}

Coordinate* Node::coordinate(CoordinateType type, Direction direction) {
    for (auto &coordinate : *_coordinates) {
        if (coordinate->type() == type && coordinate->direction() == direction) {
            return coordinate;
        }
    }
    return nullptr;
}

void Node::addCoordinate(CoordinateType type, Direction direction, double value) {
    for (auto &Coordinate : *_coordinates) {
        if (Coordinate->type() == type && Coordinate->direction() == direction) {
            throw std::invalid_argument("Trying to add a coordinate that already exists");
        }
    }
    _coordinates->push_back(new Coordinate(type, direction, value));
    return;
}

void Node::addCoordinate(CoordinateType type, Direction direction) {
    for (auto &Coordinate : *_coordinates) {
        if (Coordinate->type() == type && Coordinate->direction() == direction) {
            throw std::invalid_argument("Trying to add a coordinate that already exists");
        }
    }
    _coordinates->push_back(new Coordinate(type, direction));
}

void Node::removeCoordinate(PositioningInSpace::CoordinateType type, PositioningInSpace::Direction direction) {
    for (auto &Coordinate : *_coordinates) {
        if (Coordinate->type() == type && Coordinate->direction() == direction) {
            _coordinates->remove(Coordinate);
            delete Coordinate;
            Coordinate = nullptr;}
    }
    throw std::invalid_argument("Trying to remove a coordinate that does not exist");
}

DegreeOfFreedom* Node::degreeOfFreedom(DOFType type, FieldType fieldType) {
    for (auto &degreeOfFreedom : *_degreesOfFreedom) {
        if (degreeOfFreedom->type() == type && degreeOfFreedom->fieldType() == fieldType) {
            return degreeOfFreedom;
        }
    }
    return nullptr;
}

void Node::addDegreeOfFreedom(DOFType type, FieldType fieldType, double value) {
    for (auto &degreeOfFreedom: *_degreesOfFreedom)
        if (degreeOfFreedom->type() == type && degreeOfFreedom->fieldType() == fieldType)
            throw std::invalid_argument("Trying to add a degree of freedom that already exists");


    _degreesOfFreedom->push_back(new DegreeOfFreedom(type, fieldType, value));
}

void Node::addDegreeOfFreedom(DOFType type, FieldType fieldType) {
    for (auto &degreeOfFreedom: *_degreesOfFreedom)
        if (degreeOfFreedom->type() == type && degreeOfFreedom->fieldType() == fieldType)
            throw std::invalid_argument("Trying to add a degree of freedom that already exists");
    _degreesOfFreedom->push_back(new DegreeOfFreedom(type, fieldType));
}

void Node::removeDegreeOfFreedom(DOFType type, FieldType fieldType) {
    for (auto &degreeOfFreedom : *_degreesOfFreedom) {
        if (degreeOfFreedom->type() == type && degreeOfFreedom->fieldType() == fieldType) {
            _degreesOfFreedom->remove(degreeOfFreedom);
            delete degreeOfFreedom;
            degreeOfFreedom = nullptr;}
    }
    throw std::invalid_argument("Trying to remove a degree of freedom that does not exist");
}*/
