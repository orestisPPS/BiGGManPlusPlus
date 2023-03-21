//
// Created by hal9000 on 2/16/23.
//
#include <stdexcept>
#include <utility>
#include "BoundaryCondition.h"

namespace BoundaryConditions {
    
    BoundaryCondition::BoundaryCondition(BoundaryConditionType bcType, map<DOFType*, function<double (vector<double>*)>>* bcForDof) :
            _bcType(bcType), bcForDof(bcForDof) {
    }
    
    double BoundaryCondition::scalarValueOfDOFAt(DOFType type, vector<double> *coordinates) {
        return bcForDof->at(&type)(coordinates);
    }

    vector<double> BoundaryCondition::vectorValueOfAllDOFAt(vector<double> *coordinates) {
        vector<double> result = vector<double>(bcForDof->size());
        for (auto &bc : *bcForDof) {
            result.push_back(bc.second(coordinates));
        }        
    }
    
    const BoundaryConditionType &BoundaryCondition::type() const {
        return _bcType;
    }




} // BoundaryConditions