//
// Created by hal9000 on 1/31/23.
//

#include "DegreeOfFreedomID.h"

namespace DegreesOfFreedom {
    
    DegreeOfFreedomID::DegreeOfFreedomID(ConstraintType type) : _constraintType(type){
        value = new unsigned int;
    }
    
    DegreeOfFreedomID::~DegreeOfFreedomID() {
        if (value != nullptr)
            delete value;
        delete globalValue;
        value = nullptr;
        globalValue = nullptr;
    }
    
    bool DegreeOfFreedomID::operator == (const DegreeOfFreedomID& dof) const {
        return *value == *dof.value;
    }
    
    bool DegreeOfFreedomID::operator != (const DegreeOfFreedomID& dof) const {
        return *value != *dof.value;
    }
    
    //Constant reference to an enum that indicates whether the degree of freedom is
    // fixed (Dirichlet BC), flux (Neumann BC), or free.
    const ConstraintType& DegreeOfFreedomID::constraintType() {
        return _constraintType;
    }

} // DegreesOfFreedom