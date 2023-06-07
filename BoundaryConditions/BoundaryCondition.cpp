//
// Created by hal9000 on 2/16/23.
//
#include <stdexcept>
#include <utility>
#include "BoundaryCondition.h"

namespace BoundaryConditions {
    
    BoundaryCondition::BoundaryCondition(BoundaryConditionType bcType, map<DOFType, function<double (vector<double>*)>>* bcFunctionForDof) :
            _bcType(bcType), _bcFunctionForDof(bcFunctionForDof), _bcValueForDof(nullptr) {
    }
    
    BoundaryCondition::BoundaryCondition(BoundaryConditionType bcType, map<DOFType, double>* bcValueForDof) :
            _bcType(bcType), _bcValueForDof(bcValueForDof), _bcFunctionForDof(nullptr) {
    }
    
    double BoundaryCondition::scalarValueOfDOFAt(DOFType type, vector<double>* coordinates) {
        return _bcFunctionForDof->at(type)(coordinates);
    }
    
    double BoundaryCondition::scalarValueOfDOFAt(DOFType type) {
        return _bcValueForDof->at(type);
    }

    vector<double> BoundaryCondition::vectorValueOfAllDOFAt(vector<double> *coordinates) {
        vector<double> result = vector<double>(_bcFunctionForDof->size());
        for (auto &bc : *_bcFunctionForDof) {
            result.push_back(bc.second(coordinates));
        }
        return result;
    }
    
    const BoundaryConditionType &BoundaryCondition::type() const {
        return _bcType;
    }




} // BoundaryConditions