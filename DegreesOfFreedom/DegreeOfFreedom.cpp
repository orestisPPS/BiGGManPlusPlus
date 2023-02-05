//
// Created by hal9000 on 11/28/22.
//

#include "DegreeOfFreedom.h"
#include <iostream>
#include "limits"

namespace DegreesOfFreedom{
    DegreeOfFreedom::DegreeOfFreedom(DOFType dofType) :
    _dofType(dofType),
    _value(std::numeric_limits<double>::quiet_NaN()),
    id(){
    }
    

    DegreeOfFreedom::DegreeOfFreedom(DOFType dofType, double value) : _dofType(dofType), _value(value), id() {
    }
    
    DOFType const &DegreeOfFreedom::type() {
        return _dofType;
    }
    

    double DegreeOfFreedom::value() const {
        return _value;
    }

    void DegreeOfFreedom::setValue(double value) {
        _value = value;
    }

    bool DegreeOfFreedom::operator==(const DegreeOfFreedom &dof) {
        if (value() != numeric_limits<double>::quiet_NaN()) {
            return _dofType == dof._dofType && _value == dof._value;
        }
        return _dofType == dof._dofType;
    }

    bool DegreeOfFreedom::operator!=(const DegreeOfFreedom &dof) {
        return !(*this == dof);
    }

    void DegreeOfFreedom::Print() {
        std::cout << "DOFType: " << _dofType << " Value: " << _value <<  std::endl;
    }
}

