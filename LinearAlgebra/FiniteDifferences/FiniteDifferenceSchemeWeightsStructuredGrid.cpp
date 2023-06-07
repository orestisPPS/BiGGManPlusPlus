//
// Created by hal9000 on 1/28/23.
//

#include "FiniteDifferenceSchemeWeightsStructuredGrid.h"

namespace LinearAlgebra {
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid::
    _getSchemeFromDerivativeOrder(FDSchemeType schemeType, unsigned derivativeOrder, unsigned int errorOrder) {

        auto schemeTypeAndOrder = make_tuple(schemeType, errorOrder);
        auto schemeTypeAndOrderToSchemeWeightsMap = map<tuple<FDSchemeType, unsigned int>, SchemeMap>();
        switch (derivativeOrder) {
            case 1:
                schemeTypeAndOrderToSchemeWeightsMap = _schemeTypeAndOrderToWeightsDerivative1();
                break;
            case 2:
                schemeTypeAndOrderToSchemeWeightsMap = _schemeTypeAndOrderToWeightsDerivative2();
                break;
            default:
                throw invalid_argument("Derivative order should be 1 or 2");
        }
        if (schemeTypeAndOrderToSchemeWeightsMap.find(make_tuple(schemeType, errorOrder)) !=
            schemeTypeAndOrderToSchemeWeightsMap.end()) {
            return schemeTypeAndOrderToSchemeWeightsMap[schemeTypeAndOrder];
        } else {
            throw invalid_argument("The scheme type and error order combination is not supported");
        }
    }

    Scheme FiniteDifferenceSchemeWeightsStructuredGrid::
    getScheme(FDSchemeType schemeType, unsigned short derivativeOrder, unsigned errorOrder) {
        
        auto schemeMap = _getSchemeFromDerivativeOrder(schemeType, derivativeOrder, errorOrder);
        auto scheme = Scheme();
        scheme.power = schemeMap.power;
        scheme.denominatorCoefficient = schemeMap.denominatorCoefficient;
        scheme.weights = vector<double>();
        for (auto& weight : schemeMap.weights) {
            scheme.weights.push_back(weight.second);
        }
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_1() {
        auto weights = map<int, double>();
        weights[0] = 1;
        weights[1] = -1;
        auto scheme = SchemeMap();
        scheme.power = 1;
        scheme.denominatorCoefficient = 1.0;
        scheme.weights = weights;
        return scheme;
    }

    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_2() {
        auto weights = map<int, double>();
        weights[0] = 3.0;
        weights[1] = -4.0;
        weights[2] = 1.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 2.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_3() {
        auto weights = map<int, double>();
        weights[0] = 11.0;
        weights[1] = -18.0;
        weights[2] = 9.0;
        weights[3] = -2.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 6.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_4() {
        auto weights = map<int, double>();
        weights[0] = 25.0;
        weights[1] = -48.0;
        weights[2] = 36.0;
        weights[3] = -16.0;
        weights[4] = 3.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 12.0;
        return scheme;
    }
    
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward1_5() {
        auto weights = map<int, double>();
        weights[0] = 137.0;
        weights[1] = -300.0;
        weights[2] = 300.0;
        weights[3] = -200.0;
        weights[4] = 75.0;
        weights[5] = -12.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 60.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_1() {
        auto weights = map<int, double>();
        weights[-1] = -1;
        weights[0] = 1;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 1.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_2() {
        auto weights = map<int, double>();
        weights[-2] = 1;
        weights[-1] = -4;
        weights[0] = 3;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 2.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_3() {
        auto weights = map<int, double>();
        weights[-3] = -2;
        weights[-2] = 9;
        weights[-1] = -18;
        weights[0] = 11;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 6.0;
        return scheme;
    }
    
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_4() {
        auto weights = map<int, double>();
        weights[-4] = 3;
        weights[-3] = -16;
        weights[-2] = 36;
        weights[-1] = -48;
        weights[0] = 25;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 12.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward1_5() {
        auto weights = map<int, double>();
        weights[-5] = -12;
        weights[-4] = 75;
        weights[-3] = -200;
        weights[-2] = 300;
        weights[-1] = -300;
        weights[0] = 137;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 60.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _central1_2() {
        auto weights = map<int, double>();
        weights[-1] = -1.0;
        weights[0] = 0.0;
        weights[1] = 1.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 2.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _central1_4() {
        auto weights = map<int, double>();
        weights[-2] = 1;
        weights[-1] = -8;
        weights[0] = 0;
        weights[1] = 8;
        weights[2] = -1;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 12.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _central1_6() {
        auto weights = map<int, double>();
        weights[-3] = 1;
        weights[-2] = -9;
        weights[-1] = 45;
        weights[0] = 0;
        weights[1] = -45;
        weights[2] = 9;
        weights[3] = -1;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 1;
        scheme.denominatorCoefficient = 60.0;
        return scheme;
    }

    map<tuple<FDSchemeType, unsigned>, SchemeMap>
    FiniteDifferenceSchemeWeightsStructuredGrid::_schemeTypeAndOrderToWeightsDerivative1() {
        auto firstOrderWeights = map<tuple<FDSchemeType, unsigned>, SchemeMap>();
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
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_2() {
        auto weights = map<int, double>();
        weights[0] = 2;
        weights[1] = -5;
        weights[2] = 4;
        weights[3] = -1;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 1.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_3() {
        auto weights = map<int, double>();
        weights[0] = 35.0;
        weights[1] = -104.0;
        weights[2] = 114.0;
        weights[3] = -56.0;
        weights[4] = 11.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 12.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_4() {
        auto weights = map<int, double>();
        weights[0] = 45.0;
        weights[1] = -154.0;
        weights[2] = 214.0;
        weights[3] = -156.0;
        weights[4] = 61.0;
        weights[5] = -10.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 12.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _forward2_5() {
        auto weights = map<int, double>();
        weights[0] = 812.0;
        weights[1] = -3132.0;
        weights[2] = 5265.0;
        weights[3] = -5080.0;
        weights[4] = 2970.0;
        weights[5] = -972.0;
        weights[6] = 137.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 180.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_2() {
        auto weights = map<int, double>();
        weights[-3] = -1;
        weights[-2] = 4;
        weights[-1] = -5;
        weights[0] = 2;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 1.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_3() {
        auto weights = map<int, double>();
        weights[-4] = 11.0;
        weights[-3] = -56.0;
        weights[-2] = 114.0;
        weights[-1] = -104.0;
        weights[0] = 35.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 12.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_4() {
        auto weights = map<int, double>();
        weights[-5] = -10.0;
        weights[-4] = 61.0;
        weights[-3] = -156.0;
        weights[-2] = 214.0;
        weights[-1] = -154.0;
        weights[0] = 45.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 12.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _backward2_5() {
        auto weights = map<int, double>();
        weights[-6] = 137.0;
        weights[-5] = -972.0;
        weights[-4] = 2970.0;
        weights[-3] = -5080.0;
        weights[-2] = 5265.0;
        weights[-1] = -3132.0;
        weights[0] = 812.0;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 180.0;
        
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _central2_2() {
        auto weights = map<int, double>();
        weights[-1] = 1;
        weights[0]  = -2;
        weights[1]  = 1;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 1.0;
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _central2_4() {
        auto weights = map<int, double>();
        weights[-2] = -1;
        weights[-1] = 16;
        weights[0] = -30;
        weights[1] = 16;
        weights[2] = -1;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 12.0;
        
        return scheme;
    }
    
    SchemeMap FiniteDifferenceSchemeWeightsStructuredGrid :: _central2_6() {
        auto weights = map<int, double>();
        weights[-3] = 2;
        weights[-2] = -27;
        weights[-1] = 270;
        weights[0] = -490;
        weights[1] = 270;
        weights[2] = -27;
        weights[3] = 2;
        
        auto scheme = SchemeMap();
        scheme.weights = weights;
        scheme.power = 2;
        scheme.denominatorCoefficient = 180.0;
        
        return scheme;
    }
    
    map<tuple<FDSchemeType, unsigned>, SchemeMap> FiniteDifferenceSchemeWeightsStructuredGrid :: _schemeTypeAndOrderToWeightsDerivative2() {
        auto secondOrderWeights = map<tuple<FDSchemeType, unsigned>, SchemeMap>();
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