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
            
        _timeDependentValues = nullptr;
        _firstTemporalDerivativeValues = nullptr;
        _secondTemporalDerivativeValues = nullptr;
    }
    
    bool DegreeOfFreedom::operator==(const DegreeOfFreedom &dof) {
        return _dofType == dof._dofType &&
               _parentNode == dof._parentNode &&
                _constraintType == dof._constraintType &&
                _value == dof._value &&
                _timeDependentValues == dof._timeDependentValues &&
                _firstTemporalDerivativeValues == dof._firstTemporalDerivativeValues &&
                _secondTemporalDerivativeValues == dof._secondTemporalDerivativeValues;
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
    
    double const DegreeOfFreedom::value() const{
        return _value;
    }
    
    double const DegreeOfFreedom::value(unsigned derivativeOrder, unsigned stepIndex) const {
        switch (derivativeOrder) {
            case 0:
                if (_timeDependentValues == nullptr)
                    return _value;
                return _timeDependentValues->at(stepIndex);
            case 1:
                if (_firstTemporalDerivativeValues == nullptr)
                    return 0;
                return _firstTemporalDerivativeValues->at(stepIndex);
            case 2:
                if (_secondTemporalDerivativeValues == nullptr)
                    return 0;
                return _secondTemporalDerivativeValues->at(stepIndex);
            default:
                throw std::invalid_argument("Temporal derivative order should be 0, 1 or 2");
        }
    }

    void DegreeOfFreedom::setValue(double value) {
        _value = value;
    }
    
    void DegreeOfFreedom::setValue(unique_ptr<NumericalVector<double>> result, unsigned temporalDerivativeOrder) {
        switch (temporalDerivativeOrder) {
            case 0:
                _timeDependentValues = std::move(result);
                break;
            case 1:
                _firstTemporalDerivativeValues = std::move(result);
                break;
            case 2:
                _secondTemporalDerivativeValues = std::move(result);
                break;
            default:
                throw std::invalid_argument("Temporal derivative order should be 0, 1 or 2");
        }
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
