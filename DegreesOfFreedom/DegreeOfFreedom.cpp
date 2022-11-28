//
// Created by hal9000 on 11/28/22.
//

#include "DegreeOfFreedom.h"
#include <iostream>
#include "limits"

DegreeOfFreedom::DegreeOfFreedom(DOFType dofType, FieldType fieldType) {
    _dofType = dofType;
    _fieldType = fieldType;
    _value = new double(std::numeric_limits<double>::quiet_NaN());
}

DegreeOfFreedom::DegreeOfFreedom(DOFType dofType, FieldType fieldType, double value) {
    _dofType = dofType;
    _fieldType = fieldType;
    _value = new double(value);
}

DegreeOfFreedom::~DegreeOfFreedom() {
    delete _value;
    _value = nullptr;
}

DOFType DegreeOfFreedom::type() {
    return _dofType;
}

FieldType DegreeOfFreedom::fieldType() {
    return _fieldType;
}

double DegreeOfFreedom::value() {
    return *_value;
}

void DegreeOfFreedom::setValue(double value) {
    *_value = value;
}

bool DegreeOfFreedom::operator==(const DegreeOfFreedom &dof) {
    switch (*_value != std::numeric_limits<double> ::quiet_NaN()) {
        case true:
            return _dofType == dof._dofType && _fieldType == dof._fieldType && *_value == *dof._value;
        case false:
            return _dofType == dof._dofType && _fieldType == dof._fieldType;
    }
    return false;
}

bool DegreeOfFreedom::operator!=(const DegreeOfFreedom &dof) {
    return !(*this == dof);
}

void DegreeOfFreedom::Print() {
    std::cout << "DOFType: " << _dofType << " FieldType: " << _fieldType << " Value: " << *_value <<  std::endl;
}

