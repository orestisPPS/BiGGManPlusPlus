//
// Created by hal9000 on 4/23/23.
//

#ifndef UNTITLED_FINITEDIFFERENCESCHEMEWEIGHTSCALCULATOR_H
#define UNTITLED_FINITEDIFFERENCESCHEMEWEIGHTSCALCULATOR_H
#include <cmath>
#include "../../Utility/Exporters/Exporters.h"
#include "../LinearSystem.h"
#include "../Solvers/Direct/SolverLUP.h"
#include "../Operations/NumberOperations.h"

using namespace std;

namespace LinearAlgebra {

    class FiniteDifferenceSchemeWeightsCalculator {
    public:

        static vector<double> calculateVandermondeCoefficients(unsigned derivativeOrder, vector<double>& positions);
        

    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEWEIGHTSCALCULATOR_H
