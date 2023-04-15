//
// Created by hal9000 on 1/28/23.
//

#include "FirstOrderDerivativeFDSchemeCalculator.h"

namespace LinearAlgebra {

    FirstOrderDerivativeFDSchemeCalculator::FirstOrderDerivativeFDSchemeCalculator(){
        
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator::
    getSchemeValues(FiniteDifferenceSchemeType schemeType, unsigned errorOrder, map<int, double>& functionValues, double stepSize) {
        auto weights = getWeightsOfSchemeTypeAndError(schemeType, errorOrder);
        if (weights.size() != functionValues.size()) {
            throw invalid_argument("The number of weights and function values must be the same!");
        }
        for (auto weight : weights) {
            auto index = weight.first;
            if (functionValues.find(index) == functionValues.end()) {
                throw invalid_argument("The function values must contain the index of the weight!");
            }
        }
        auto schemeValues = map<int, double>();
        for (auto weight : weights) {
            auto index = weight.first;
            auto weightValue = weight.second;
            schemeValues[index] = weightValue * functionValues[index] / stepSize;
        }
        return schemeValues;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator::
    getWeightsOfSchemeTypeAndError(FiniteDifferenceSchemeType schemeType, unsigned int errorOrder) {
        auto schemeTypeAndOrder = make_tuple(schemeType, errorOrder);
        auto schemeTypeAndOrderToSchemeWeightsMap = _schemeTypeAndOrderToSchemeWeights();
        if (schemeTypeAndOrderToSchemeWeightsMap.find(schemeTypeAndOrder) != schemeTypeAndOrderToSchemeWeightsMap.end()) {
            return schemeTypeAndOrderToSchemeWeightsMap[schemeTypeAndOrder];
        } else {
            throw invalid_argument("The scheme type and error order combination is not supported");
        }
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward1() {
        auto weights = map<int, double>();
        weights[0] = 1;
        weights[1] = -1;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward2() {
        auto weights = map<int, double>();
        weights[0] = 3;
        weights[1] = -4;
        weights[2] = 1;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward3() {
        auto weights = map<int, double>();
        weights[0] = 11;
        weights[1] = -18;
        weights[2] = 9;
        weights[3] = -2;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward4() {
        auto weights = map<int, double>();
        weights[0] = 25;
        weights[1] = -48;
        weights[2] = 36;
        weights[3] = -16;
        weights[4] = 3;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward5() {
        auto weights = map<int, double>();
        weights[0] = 137;
        weights[1] = -300;
        weights[2] = 300;
        weights[3] = -200;
        weights[4] = 75;
        weights[5] = -12;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward1() {
        auto weights = map<int, double>();
        weights[0] = 1;
        weights[-1] = -1;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward2() {
        auto weights = map<int, double>();
        weights[0] = 3;
        weights[-1] = -4;
        weights[-2] = 1;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward3() {
        auto weights = map<int, double>();
        weights[0] = 11;
        weights[-1] = -18;
        weights[-2] = 9;
        weights[-3] = -2;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward4() {
        auto weights = map<int, double>();
        weights[0] = 25;
        weights[-1] = -48;
        weights[-2] = 36;
        weights[-3] = -16;
        weights[-4] = 3;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward5() {
        auto weights = map<int, double>();
        weights[0] = 137;
        weights[-1] = -300;
        weights[-2] = 300;
        weights[-3] = -200;
        weights[-4] = 75;
        weights[-5] = -12;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _central2() {
        auto weights = map<int, double>();
        weights[1] = -1;
        weights[0] = 1;
        weights[-1] = -1;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _central4() {
        auto weights = map<int, double>();
        weights[2] = -1;
        weights[1] = 8;
        weights[0] = -8;
        weights[-1] = 1;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _central6() {
        auto weights = map<int, double>();
        weights[3] = -1;
        weights[2] = 9;
        weights[1] = -45;
        weights[0] = 45;
        weights[-1] = -9;
        weights[-2] = 1;
        return weights;
    }

    map<tuple<FiniteDifferenceSchemeType, unsigned>, map<int, double>>
    FirstOrderDerivativeFDSchemeCalculator::_schemeTypeAndOrderToSchemeWeights() {
        auto firstOrderWeights = map<tuple<FiniteDifferenceSchemeType, unsigned>, map<int, double>>();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Forward, 1)] = _forward1();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Forward, 2)] = _forward2();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Forward, 3)] = _forward3();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Forward, 4)] = _forward4();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Forward, 5)] = _forward5();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Backward, 1)] = _backward1();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Backward, 2)] = _backward2();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Backward, 3)] = _backward3();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Backward, 4)] = _backward4();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Backward, 5)] = _backward5();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Central, 2)] = _central2();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Central, 4)] = _central4();
        firstOrderWeights[make_tuple(FiniteDifferenceSchemeType::Central, 6)] = _central6();
        return firstOrderWeights;
    }
    


} // LinearAlgebra