//
// Created by hal9000 on 1/28/23.
//

#include "FirstOrderDerivativeFDSchemeCalculator.h"

namespace LinearAlgebra {

    FirstOrderDerivativeFDSchemeCalculator::FirstOrderDerivativeFDSchemeCalculator(){
        
    }
    
    FDSchemeComponent FirstOrderDerivativeFDSchemeCalculator::
    getScheme(FDSchemeType schemeType, unsigned errorOrder, map<int, double>* functionValues, double stepSize) {
        
        auto weights = getWeights(schemeType, errorOrder);
        
        if (weights->size() != functionValues->size())
            throw invalid_argument("The number of weights and function values must be the same!");
        
        for (auto& weight : *weights) {;
            if (functionValues->find(weight.first) == functionValues->end())
                throw invalid_argument("The function values must contain the index of the weight!");
        }
        
        return {schemeType, errorOrder, functionValues, weights, stepSize};
    }
    
    FDSchemeComponent FirstOrderDerivativeFDSchemeCalculator::
    getSchemeFromGivenPoints(map<int, double> *functionValues,double stepSize) {
        
        auto points = vector<int>();
        for (auto& functionValue : *functionValues) {
            points.push_back(functionValue.first);
        }

        //central2
        auto targetVector = vector<int>() = {-1, 0, 1};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Central, 2, functionValues, stepSize);

        //central4
        targetVector = vector<int>() = {-2, -1, 0, 1, 2};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Central, 4, functionValues, stepSize);

        //central6
        targetVector = vector<int>() = {-3, -2, -1, 0, 1, 2, 3};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Central, 6, functionValues, stepSize);
        
        //backward1
        targetVector = vector<int>() = {-1, 0};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Backward, 1, functionValues, stepSize);
        
        //backward2
        targetVector = vector<int>() = {-2, -1, 0};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Backward, 2, functionValues, stepSize);
        
        //backward3
        targetVector = vector<int>() = {-3, -2, -1, 0};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Backward, 3, functionValues, stepSize);
        
        //backward4
        targetVector = vector<int>() = {-4, -3, -2, -1, 0};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Backward, 4, functionValues, stepSize);
        
        //backward5
        targetVector = vector<int>() = {-5, -4, -3, -2, -1, 0};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Backward, 5, functionValues, stepSize);
        
        //forward1
        targetVector = vector<int>() = {0, 1};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Forward, 1, functionValues, stepSize);
        
        //forward2
        targetVector = vector<int>() = {0, 1, 2};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Forward, 2, functionValues, stepSize);
        
        //forward3
        targetVector = vector<int>() = {0, 1, 2, 3};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Forward, 3, functionValues, stepSize);
        
        //forward4
        targetVector = vector<int>() = {0, 1, 2, 3, 4};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Forward, 4, functionValues, stepSize);
        
        //forward5
        targetVector = vector<int>() = {0, 1, 2, 3, 4, 5};
        if (VectorOperations::areEqualVectors(points, targetVector))
            return getScheme(Forward, 5, functionValues, stepSize);
        

        throw invalid_argument("The given points do not correspond to any supported scheme!");
    }
    
    map<int, double>* FirstOrderDerivativeFDSchemeCalculator::
    getWeights(FDSchemeType schemeType, unsigned int errorOrder) {
        
        auto schemeTypeAndOrder = make_tuple(schemeType, errorOrder);
        auto schemeTypeAndOrderToSchemeWeightsMap = _schemeTypeAndOrderToSchemeWeights();
        if (schemeTypeAndOrderToSchemeWeightsMap.find(schemeTypeAndOrder) != schemeTypeAndOrderToSchemeWeightsMap.end()) {
            return new map<int, double>(schemeTypeAndOrderToSchemeWeightsMap[schemeTypeAndOrder]);
        }
        else {
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
        weights[0] = 3.0 / 2.0;
        weights[1] = -4 / 2.0;
        weights[2] = 1 / 2.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward3() {
        auto weights = map<int, double>();
        weights[0] = 11 / 6.0;
        weights[1] = -18 / 6.0;
        weights[2] = 9 / 6.0;
        weights[3] = -2 / 6.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward4() {
        auto weights = map<int, double>();
        weights[0] = 25 / 12.0;
        weights[1] = -48 / 6.0;
        weights[2] = 36 / 6.0;
        weights[3] = -16 / 6.0;
        weights[4] = 3 / 6.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _forward5() {
        auto weights = map<int, double>();
        weights[0] = 137 / 60.0;
        weights[1] = -300 / 60.0;
        weights[2] = 300 / 60.0;
        weights[3] = -200 / 60.0;
        weights[4] = 75 / 60.0;
        weights[5] = -12 / 60.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward1() {
        auto weights = map<int, double>();
        weights[-1] = 1;
        weights[0] = 1;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward2() {
        auto weights = map<int, double>();
        weights[-2] = 1 / 2.0;
        weights[-1] = -4 / 2.0;
        weights[0] = 3 / 2.0;


        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward3() {
        auto weights = map<int, double>();
        weights[-3] = -2 / 6.0;
        weights[-2] = 9 / 6.0;
        weights[-1] = -18 / 6.0;
        weights[0] = 11 / 6.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward4() {
        auto weights = map<int, double>();
        weights[-4] = 3 / 12.0;
        weights[-3] = -16 / 12.0;
        weights[-2] = 36 / 12.0;
        weights[-1] = -48 / 12.0;
        weights[0] = 25 / 12.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _backward5() {
        auto weights = map<int, double>();
        weights[-5] = -12 / 60.0;
        weights[-4] = 75 / 60.0;
        weights[-3] = -200 / 60.0;
        weights[-2] = 300 / 60.0;
        weights[-1] = -300 / 60.0;
        weights[0] = 137 / 60.0;

        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _central2() {
        auto weights = map<int, double>();
        weights[-1] = -1 / 2.0;
        weights[0] = 0.0;
        weights[1] = 1 / 2.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _central4() {
        auto weights = map<int, double>();
        weights[-2] = 1 / 12.0;
        weights[-1] = -8 / 12.0;
        weights[0] = 0.0;
        weights[1] = 8 / 12.0;
        weights[2] = -1 / 12.0;
        return weights;
    }
    
    map<int, double> FirstOrderDerivativeFDSchemeCalculator :: _central6() {
        auto weights = map<int, double>();
        weights[-3] = 1 / 60.0;
        weights[-2] = -9 / 60.0;
        weights[-1] = 45 / 60.0;
        weights[0] = 0.0;
        weights[1] = -45 / 60.0;
        weights[2] = 9 / 60.0;
        weights[3] = -1 / 60.0;
        return weights;
    }

    map<tuple<FDSchemeType, unsigned>, map<int, double>>
    FirstOrderDerivativeFDSchemeCalculator::_schemeTypeAndOrderToSchemeWeights() {
        auto firstOrderWeights = map<tuple<FDSchemeType, unsigned>, map<int, double>>();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 1)] = _forward1();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 2)] = _forward2();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 3)] = _forward3();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 4)] = _forward4();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 5)] = _forward5();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 1)] = _backward1();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 2)] = _backward2();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 3)] = _backward3();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 4)] = _backward4();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 5)] = _backward5();
        firstOrderWeights[make_tuple(FDSchemeType::Central, 2)] = _central2();
        firstOrderWeights[make_tuple(FDSchemeType::Central, 4)] = _central4();
        firstOrderWeights[make_tuple(FDSchemeType::Central, 6)] = _central6();
        return firstOrderWeights;
    }
    


} // LinearAlgebra