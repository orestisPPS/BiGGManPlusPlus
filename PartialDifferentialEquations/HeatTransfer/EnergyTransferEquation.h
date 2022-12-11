//
// Created by hal9000 on 12/9/22.
//

#ifndef UNTITLED_ENERGYTRANSFEREQUATION_H
#define UNTITLED_ENERGYTRANSFEREQUATION_H


#include <vector>
#include "../PartialDifferentialEquation.h"
using namespace PartialDifferentialEquations;


namespace PartialDifferentialEquations {

    class EnergyTransferEquation : public PartialDifferentialEquation {
    public:
        EnergyTransferEquation(double *density, double *specialHeatCapacity,
                               double *thermalConductivity, vector<double> *flowVelocity,
                               double *reaction double *source);
        EnergyTransferEquation(double *density, double *specialHeatCapacity,
                               Matrix<double> *anisotropicThermalConductivity, vector<double> *flowVelocity,
                               double *reaction double *source);
        EnergyTransferEquation(map<int*, Matrix<double>> localDensity, double *specialHeatCapacity,
                               map<int*, Matrix<double>*> *locallyAnisotropicThermalConductivity, map<int*, vector<double>*> *localFlowVelocity,
                               map<int*,double*> *localReaction, map<int*,double*> *localSource);
        ~EnergyTransferEquation();
    };
}
#endif //UNTITLED_ENERGYTRANSFEREQUATION_H
