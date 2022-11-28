//
// Created by hal9000 on 11/28/22.
//

#include "Coordinate.h"
#include <iostream>
#include <limits>

Coordinate::Coordinate(CoordinateType type, Direction direction) {
    _type = type;
    _direction = direction;
    _value = new double(std::numeric_limits<double>::quiet_NaN());
}

Coordinate::Coordinate(CoordinateType type, Direction direction, double value) {
    _type = type;
    _direction = direction;
    _value = new double(value);
}

Coordinate::~Coordinate() {
    delete _value;
    _value = nullptr;
}

CoordinateType Coordinate::type() {
    return _type;
}

Direction Coordinate::direction() {
    return _direction;
}

double Coordinate::value() {
    return *_value;
}

void Coordinate::setValue(double value) {
    *_value = value;
}

bool Coordinate::operator==(const Coordinate &dof) {
    switch (*_value != std::numeric_limits<double> ::quiet_NaN()) {
        case true:
            return _type == dof._type && _direction == dof._direction && *_value == *dof._value;
        case false:
            return _type == dof._type && _direction == dof._direction;
    }
    return false;
}

bool Coordinate::operator!=(const Coordinate &dof) {
    return !(*this == dof);
}

void Coordinate::Print() {
    std::cout << "CoordinateType: " << _type << " Direction: " << _direction << " Value: " << *_value <<  std::endl;
}

