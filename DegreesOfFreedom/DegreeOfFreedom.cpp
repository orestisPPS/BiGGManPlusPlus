//
// Created by hal9000 on 11/28/22.
//

#include "DegreeOfFreedom.h"

namespace DegreesOfFreedom{

    DegreeOfFreedom::DegreeOfFreedom(DOFType dofType, DegreeOfFreedomID *id, unsigned* parentNodeGlobalId) :
            _dofType(dofType), _value(numeric_limits<double>::quiet_NaN()) {
        this->id = id;
        this->parentNodeGlobalId = parentNodeGlobalId;
    }

    DegreeOfFreedom::DegreeOfFreedom(DOFType dofType, double value, DegreeOfFreedomID *id, unsigned* parentNodeGlobalId) :
            _dofType(dofType), _value(value) {
        if (id->constraintType() != ConstraintType::Fixed)
            throw std::invalid_argument("Cannot initialize a DOF with a value if it is not fixed"
                                        "Use the constructor that does not take a value");
        this->id = id;
        this->parentNodeGlobalId = parentNodeGlobalId;
    }

    DegreeOfFreedom::~DegreeOfFreedom() {
        delete id;
        delete parentNodeGlobalId;
        id = nullptr;
        parentNodeGlobalId = nullptr;
    }

    bool DegreeOfFreedom::operator==(const DegreeOfFreedom &dof) {
        return *id == *dof.id &&
               _dofType == dof._dofType &&
               _value == dof._value &&
               *parentNodeGlobalId == *dof.parentNodeGlobalId;
    }

    bool DegreeOfFreedom::operator!=(const DegreeOfFreedom &dof) {
        return !(*this == dof);
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
    //Prints in the CLI the DOF ID value, DOF type  and constraint type
    void DegreeOfFreedom::Print() {
        std::cout
                << "DOF ID: " << *id->id << " DOF Type: " << _dofType << " Constraint Type: " << id->constraintType()
                << std::endl;
    }
}
