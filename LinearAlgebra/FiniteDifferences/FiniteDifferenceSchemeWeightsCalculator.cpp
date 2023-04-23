//
// Created by hal9000 on 4/23/23.
//

#include "FiniteDifferenceSchemeWeightsCalculator.h"

namespace LinearAlgebra {
    FiniteDifferenceSchemeWeightsCalculator::
    FiniteDifferenceSchemeWeightsCalculator(){
        
    }
    
    vector<double> FiniteDifferenceSchemeWeightsCalculator::calculateWeights(unsigned derivativeOrder, vector<double>& positions) {
        auto weights = vector<double>();
        auto numberOfPoints = positions.size();

        
        auto A = Array<double>(numberOfPoints, numberOfPoints);
        auto b = vector<double>(numberOfPoints, 0);

        
        // March through all rows
        for (auto row = 0; row < numberOfPoints; row++) {
            // March through all columns
            for (auto column = 0; column < numberOfPoints; column++) {
                A(row, column) =  pow(positions[column], row);
            }
        }
        b[derivativeOrder] = 1.0;
        auto fileNameMatlab = "linearSystem.m";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        Utility::Exporters::exportLinearSystemToMatlabFile(A, b, filePath, fileNameMatlab, false);
        
        return weights;
    }
} // LinearAlgebra