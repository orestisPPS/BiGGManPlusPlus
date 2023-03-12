//
// Created by hal9000 on 1/31/23.
//

#include "DegreeOfFreedomID.h"

namespace DegreesOfFreedom {
    
    DegreeOfFreedomID::DegreeOfFreedomID(ConstraintType type) : _constraintType(type){
        id = new unsigned int;
    }
    
    DegreeOfFreedomID::~DegreeOfFreedomID() {
        delete id;
        id = nullptr;
    }
    
    bool DegreeOfFreedomID::operator == (const DegreeOfFreedomID& dof) const {
        return *id == *dof.id;
    }
    
    bool DegreeOfFreedomID::operator != (const DegreeOfFreedomID& dof) const {
        return *id != *dof.id;
    }
    
    //Constant reference to an enum that indicates whether the degree of freedom is
    // fixed (Dirichlet BC), flux (Neumann BC), or free.
    const ConstraintType& DegreeOfFreedomID::constraintType() {
        return _constraintType;
    }

} // DegreesOfFreedom