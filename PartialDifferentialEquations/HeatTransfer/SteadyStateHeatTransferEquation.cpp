//
// Created by hal9000 on 12/9/22.
//

#include "EnergyTransferEquation.h"

namespace PartialDifferentialEquations {
    EnergyTransferEquation::EnergyTransferEquation(double *density, double *specialHeatCapacity,
                                                   double *thermalConductivity, vector<double> *flowVelocity,
                                                   double *reaction) {
        _type = EnergyTransfer;
        auto dimensions = flowVelocity->size();
        auto *thermalConductivityMatrix = new Array<double>(4, 4);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (i == j && i < dimensions) {
                    thermalConductivityMatrix->populateElement(i, j, *thermalConductivity);
                } else {
                    thermalConductivityMatrix->populateElement(i, j, 0.0);
                } 

            }
            
            
        }
            
        }
    }
} // 