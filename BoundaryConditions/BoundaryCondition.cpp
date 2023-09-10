//
// Created by hal9000 on 2/16/23.
//
#include <stdexcept>
#include <utility>
#include "BoundaryCondition.h"

namespace BoundaryConditions {
    
    BoundaryCondition::BoundaryCondition(BoundaryConditionType bcType,
                                         shared_ptr<map<DOFType, function<double (shared_ptr<NumericalVector<double>>)>>> bcForDof) :
            _bcType(bcType), _bcFunctionForDof(std::move(bcForDof)), _bcValueForDof(nullptr) {
    }
    
    BoundaryCondition::BoundaryCondition(BoundaryConditionType bcType, shared_ptr<map<DOFType, double>> bcValueForDof) :
            _bcType(bcType), _bcValueForDof(std::move(bcValueForDof)), _bcFunctionForDof(nullptr) {
    }
    
    BoundaryCondition::~BoundaryCondition() {
        _bcFunctionForDof.reset();
        _bcValueForDof.reset();
    }
    
    double BoundaryCondition::getBoundaryConditionValueAtCoordinates(DOFType type, const shared_ptr<NumericalVector<double>> &coordinates) {
        return _bcFunctionForDof->at(type)(coordinates);
    }
    
    double BoundaryCondition::getBoundaryConditionValue(DOFType type) {
        return _bcValueForDof->at(type);
    }

    NumericalVector<double> BoundaryCondition::getAllBoundaryConditionValuesAtCoordinates(const shared_ptr<NumericalVector< double>>&coordinates) {
        NumericalVector<double> result = NumericalVector<double>(_bcFunctionForDof->size());
        auto i = 0;
        for (auto &bcFunction : *_bcFunctionForDof) {
            result[i] = bcFunction.second(coordinates);
            i++;
        }
        return result;
    }
    
    const BoundaryConditionType &BoundaryCondition::type() const {
        return _bcType;
    }




} // BoundaryConditions