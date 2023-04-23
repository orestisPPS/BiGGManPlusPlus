//
// Created by hal9000 on 4/23/23.
//

#ifndef UNTITLED_FINITEDIFFERENCESCHEMEWEIGHTSCALCULATOR_H
#define UNTITLED_FINITEDIFFERENCESCHEMEWEIGHTSCALCULATOR_H
#include <cmath>
#include "../../Utility/Exporters/Exporters.h"

using namespace std;

namespace LinearAlgebra {

    class FiniteDifferenceSchemeWeightsCalculator {
    public:
        FiniteDifferenceSchemeWeightsCalculator();
        
        static vector<double> calculateWeights(unsigned derivativeOrder, vector<double>& positions);

    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEWEIGHTSCALCULATOR_H
