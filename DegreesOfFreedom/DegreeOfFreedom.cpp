//
// Created by hal9000 on 11/28/22.
//

#include "DegreeOfFreedom.h"


namespace DegreesOfFreedom{

    DegreeOfFreedom::DegreeOfFreedom(DOFType* dofType, unsigned* parentNode, bool isConstrained, double value) :
       _dofType(dofType), _parentNode(parentNode), _value(value), _id(numeric_limits<unsigned int>::quiet_NaN()) {
        if (isConstrained)
            _constraintType = Fixed;
        else
            _constraintType = Free;
            
    }
    
    bool DegreeOfFreedom::operator==(const DegreeOfFreedom &dof) {
        return _dofType == dof._dofType &&
               _parentNode == dof._parentNode;
    }

    bool DegreeOfFreedom::operator!=(const DegreeOfFreedom &dof) {
        return !(*this == dof);
    }
    
    unsigned int const &DegreeOfFreedom::ID() const{
        return _id;
    }

    DOFType const &DegreeOfFreedom::type() const{
        return *(_dofType);
    }

    ConstraintType const &DegreeOfFreedom::constraintType() const{
        return _constraintType;
    }

    unsigned int* const &DegreeOfFreedom::parentNode() const {
        return _parentNode;
    }
    
    double const &DegreeOfFreedom::value() const{
        return _value;
    }

    void DegreeOfFreedom::setValue(double value) {
        _value = value;
    }
    
    void DegreeOfFreedom::setID(unsigned int ID) {
        _id = ID;
    }

    void DegreeOfFreedom::print(bool printValue) {
        if (printValue)
            cout << "DOF ID: " << _id << "Parent Node: " << _parentNode<< " DOF type: " << *_dofType << " Constraint type: " << _constraintType << " Value: " << _value << endl;
        else
            cout << "DOF ID: " << _id << "Parent Node: " << _parentNode<< " DOF type: " << *_dofType << " Constraint type: " << _constraintType << endl;
    }


}
