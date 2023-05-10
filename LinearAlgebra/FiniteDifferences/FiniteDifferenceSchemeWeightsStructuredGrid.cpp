//
// Created by hal9000 on 1/28/23.
//

#include "FiniteDifferenceSchemeWeightsStructuredGrid.h"

namespace LinearAlgebra {
    
    
    map<int, double>* FiniteDifferenceSchemeWeightsStructuredGrid::
    _getWeightsDerivative1(FDSchemeType schemeType, unsigned int errorOrder) {
        
        auto schemeTypeAndOrder = make_tuple(schemeType, errorOrder);
        auto schemeTypeAndOrderToSchemeWeightsMap = _schemeTypeAndOrderToWeightsDerivative1();
        if (schemeTypeAndOrderToSchemeWeightsMap.find(schemeTypeAndOrder) != schemeTypeAndOrderToSchemeWeightsMap.end()) {
            return new map<int, double>(schemeTypeAndOrderToSchemeWeightsMap[schemeTypeAndOrder]);
        }
        else {
            throw invalid_argument("The scheme type and error order combination is not supported");
        }
    }

    map<int, double>* FiniteDifferenceSchemeWeightsStructuredGrid::
    _getWeightsDerivative2(FDSchemeType schemeType, unsigned int errorOrder) {

        auto schemeTypeAndOrder = make_tuple(schemeType, errorOrder);
        auto schemeTypeAndOrderToSchemeWeightsMap = _schemeTypeAndOrderToWeightsDerivative2();
        if (schemeTypeAndOrderToSchemeWeightsMap.find(schemeTypeAndOrder) != schemeTypeAndOrderToSchemeWeightsMap.end()) {
            return new map<int, double>(schemeTypeAndOrderToSchemeWeightsMap[schemeTypeAndOrder]);
        }
        else {
            throw invalid_argument("The scheme type and error order combination is not supported");
        }
    }
    
    vector<double> FiniteDifferenceSchemeWeightsStructuredGrid::getWeightsVector(unsigned short derivativeOrder, FDSchemeType schemeType, unsigned int errorOrder) {
        if (derivativeOrder == 1) {
            return getWeightsVectorDerivative1(schemeType, errorOrder);
        }
        else if (derivativeOrder == 2) {
            return getWeightsVectorDerivative2(schemeType, errorOrder);
        }
        else {
            throw invalid_argument("The derivative order is not supported");
        }
    }

    vector<double> FiniteDifferenceSchemeWeightsStructuredGrid::getWeightsVectorDerivative1(FDSchemeType schemeType, unsigned int errorOrder) {
        auto weights = _getWeightsDerivative1(schemeType, errorOrder);
        auto weightsVector = vector<double>();
        for (auto& weight : *weights) {
            weightsVector.push_back(weight.second);
        }
        return weightsVector;
    }

    vector<double> FiniteDifferenceSchemeWeightsStructuredGrid::getWeightsVectorDerivative2(FDSchemeType schemeType, unsigned int errorOrder) {
        auto weights = _getWeightsDerivative2(schemeType, errorOrder);
        auto weightsVector = vector<double>();
        for (auto& weight : *weights) {
            weightsVector.push_back(weight.second);
        }
        return weightsVector;
    }
    
    
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_1() {
        auto weights = map<int, double>();
        weights[0] = 1;
        weights[1] = -1;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_2() {
        auto weights = map<int, double>();
        weights[0] = 3.0 / 2.0;
        weights[1] = -4 / 2.0;
        weights[2] = 1 / 2.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_3() {
        auto weights = map<int, double>();
        weights[0] = 11 / 6.0;
        weights[1] = -18 / 6.0;
        weights[2] = 9 / 6.0;
        weights[3] = -2 / 6.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_4() {
        auto weights = map<int, double>();
        weights[0] = 25 / 12.0;
        weights[1] = -48 / 6.0;
        weights[2] = 36 / 6.0;
        weights[3] = -16 / 6.0;
        weights[4] = 3 / 6.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_5() {
        auto weights = map<int, double>();
        weights[0] = 137 / 60.0;
        weights[1] = -300 / 60.0;
        weights[2] = 300 / 60.0;
        weights[3] = -200 / 60.0;
        weights[4] = 75 / 60.0;
        weights[5] = -12 / 60.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_1() {
        auto weights = map<int, double>();
        weights[-1] = 1;
        weights[0] = 1;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_2() {
        auto weights = map<int, double>();
        weights[-2] = 1 / 2.0;
        weights[-1] = -4 / 2.0;
        weights[0] = 3 / 2.0;


        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_3() {
        auto weights = map<int, double>();
        weights[-3] = -2 / 6.0;
        weights[-2] = 9 / 6.0;
        weights[-1] = -18 / 6.0;
        weights[0] = 11 / 6.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_4() {
        auto weights = map<int, double>();
        weights[-4] = 3 / 12.0;
        weights[-3] = -16 / 12.0;
        weights[-2] = 36 / 12.0;
        weights[-1] = -48 / 12.0;
        weights[0] = 25 / 12.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_5() {
        auto weights = map<int, double>();
        weights[-5] = -12 / 60.0;
        weights[-4] = 75 / 60.0;
        weights[-3] = -200 / 60.0;
        weights[-2] = 300 / 60.0;
        weights[-1] = -300 / 60.0;
        weights[0] = 137 / 60.0;

        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _central1_2() {
        auto weights = map<int, double>();
        weights[-1] = -1 / 2.0;
        weights[0] = 0.0;
        weights[1] = 1 / 2.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _central1_4() {
        auto weights = map<int, double>();
        weights[-2] = 1 / 12.0;
        weights[-1] = -8 / 12.0;
        weights[0] = 0.0;
        weights[1] = 8 / 12.0;
        weights[2] = -1 / 12.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _central1_6() {
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
    FiniteDifferenceSchemeWeightsStructuredGrid::_schemeTypeAndOrderToWeightsDerivative1() {
        auto firstOrderWeights = map<tuple<FDSchemeType, unsigned>, map<int, double>>();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 1)] = _forward1_1();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 2)] = _forward1_2();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 3)] = _forward1_3();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 4)] = _forward1_4();
        firstOrderWeights[make_tuple(FDSchemeType::Forward, 5)] = _forward1_5();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 1)] = _backward1_1();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 2)] = _backward1_2();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 3)] = _backward1_3();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 4)] = _backward1_4();
        firstOrderWeights[make_tuple(FDSchemeType::Backward, 5)] = _backward1_5();
        firstOrderWeights[make_tuple(FDSchemeType::Central, 2)] = _central1_2();
        firstOrderWeights[make_tuple(FDSchemeType::Central, 4)] = _central1_4();
        firstOrderWeights[make_tuple(FDSchemeType::Central, 6)] = _central1_6();
        return firstOrderWeights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_2() {
        auto weights = map<int, double>();
        weights[0] = 2;
        weights[1] = -5;
        weights[2] = 4;
        weights[3] = -1;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_3() {
        auto weights = map<int, double>();
        weights[0] = 35 / 12.0;
        weights[1] = -104 / 12.0;
        weights[2] = 114 / 12.0;
        weights[3] = -56 / 12.0;
        weights[4] = 11 / 12.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_4() {
        auto weights = map<int, double>();
        weights[0] =  45  / 12.0;
        weights[1] = -154 / 12.0;
        weights[2] =  214 / 12.0;
        weights[3] = -156 / 12.0;
        weights[4] =  61  / 12.0;
        weights[5] = -10  / 12.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_5() {
        auto weights = map<int, double>();
        weights[0] = 812 / 180.0;
        weights[1] = -3132 / 180.0;
        weights[2] = 5265 / 180.0;
        weights[3] = -5080 / 180.0;
        weights[4] = -2970 / 180.0;
        weights[5] = -972 / 180.0;
        weights[6] = 137 / 180.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_2() {
        auto weights = map<int, double>();
        weights[-3] = 1;
        weights[-2] = -4;
        weights[-1] = 5;
        weights[0] = -2;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_3() {
        auto weights = map<int, double>();
        weights[-4] = 11 / 12.0;
        weights[-3] = -56 / 12.0;
        weights[-2] = 114 / 12.0;
        weights[-1] = -104 / 12.0;
        weights[0] = 35 / 12.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_4() {
        auto weights = map<int, double>();
        weights[-5] = -10 / 12.0;
        weights[-4] = 61 / 12.0;
        weights[-3] = -156 / 12.0;
        weights[-2] = 214 / 12.0;
        weights[-1] = -154 / 12.0;
        weights[0] = 45 / 12.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_5() {
        auto weights = map<int, double>();
        weights[-6] = 137 / 180.0;
        weights[-5] = -972 / 180.0;
        weights[-4] = 2970 / 180.0;
        weights[-3] = -5080 / 180.0;
        weights[-2] = 5265 / 180.0;
        weights[-1] = -3132 / 180.0;
        weights[0] = 812 / 180.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _central2_2() {
        auto weights = map<int, double>();
        weights[-1] =  1;
        weights[0]  = -2;
        weights[1]  =  1;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _central2_4() {
        auto weights = map<int, double>();
        weights[-2] = -1 / 12.0;
        weights[-1] = 16 / 12.0;
        weights[0] = -30 / 12.0;
        weights[1] = 16 / 12.0;
        weights[2] = -1 / 12.0;
        return weights;
    }
    
    map<int, double> FiniteDifferenceSchemeWeightsStructuredGrid :: _central2_6() {
        auto weights = map<int, double>();
        weights[-3] = 2 / 180.0;
        weights[-2] = -27 / 180.0;
        weights[-1] = 270 / 180.0;
        weights[0] = -490 / 180.0;
        weights[1] = 270 / 180.0;
        weights[2] = -27 / 180.0;
        weights[3] = 2 / 180.0;
        return weights;
    }
    
    map<tuple<FDSchemeType, unsigned>, map<int, double>> FiniteDifferenceSchemeWeightsStructuredGrid :: _schemeTypeAndOrderToWeightsDerivative2() {
        auto secondOrderWeights = map<tuple<FDSchemeType, unsigned>, map<int, double>>();
        secondOrderWeights[make_tuple(FDSchemeType::Forward, 2)] = _forward2_2();
        secondOrderWeights[make_tuple(FDSchemeType::Forward, 3)] = _forward2_3();
        secondOrderWeights[make_tuple(FDSchemeType::Forward, 4)] = _forward2_4();
        secondOrderWeights[make_tuple(FDSchemeType::Forward, 5)] = _forward2_5();
        secondOrderWeights[make_tuple(FDSchemeType::Backward, 2)] = _backward2_2();
        secondOrderWeights[make_tuple(FDSchemeType::Backward, 3)] = _backward2_3();
        secondOrderWeights[make_tuple(FDSchemeType::Backward, 4)] = _backward2_4();
        secondOrderWeights[make_tuple(FDSchemeType::Backward, 5)] = _backward2_5();
        secondOrderWeights[make_tuple(FDSchemeType::Central, 2)] = _central2_2();
        secondOrderWeights[make_tuple(FDSchemeType::Central, 4)] = _central2_4();
        secondOrderWeights[make_tuple(FDSchemeType::Central, 6)] = _central2_6();
        return secondOrderWeights;
    }
    
    

    


} // LinearAlgebra