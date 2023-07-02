//
// Created by hal9000 on 2/16/23.
//
#include <stdexcept>
#include <utility>
#include "BoundaryCondition.h"

namespace BoundaryConditions {
    
    BoundaryCondition::BoundaryCondition(BoundaryConditionType bcType,
                                         shared_ptr<map<DOFType, function<double (shared_ptr<vector<double>>)>>> bcForDof) :
            _bcType(bcType), _bcFunctionForDof(std::move(bcForDof)), _bcValueForDof(nullptr) {
    }
    
    BoundaryCondition::BoundaryCondition(BoundaryConditionType bcType, map<DOFType, double>* bcValueForDof) :
            _bcType(bcType), _bcValueForDof(bcValueForDof), _bcFunctionForDof(nullptr) {
    }
    
    double BoundaryCondition::getBoundaryConditionValueAtCoordinates(DOFType type, const shared_ptr<vector<double>> &coordinates) {
        return _bcFunctionForDof->at(type)(coordinates);
    }
    
    double BoundaryCondition::getBoundaryConditionValue(DOFType type) {
        return _bcValueForDof->at(type);
    }

    vector<double> BoundaryCondition::getAllBoundaryConditionValuesAtCoordinates(const shared_ptr<vector < double>>&coordinates) {
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