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
                               double *thermalConductivity, shared_ptr<Array<double>>flowVelocity,
                               double *reaction, double *source);
        EnergyTransferEquation(double *density, double *specialHeatCapacity,
                               shared_ptr<Array<double>>anisotropicThermalConductivity, shared_ptr<Array<double>>flowVelocity,
                               double *reaction, double *source);
        EnergyTransferEquation(map<int*, shared_ptr<Array<double>>> localDensity, double *specialHeatCapacity,
                               map<int*, shared_ptr<Array<double>>> *locallyAnisotropicThermalConductivity, map<int*, shared_ptr<vector<double>>> *localFlowVelocity,
                               map<int*,double*> *localReaction, map<int*,double*> *localSource);
        ~EnergyTransferEquation();
        
        shared_ptr<SecondOrderLinearPDEProperties>properties;
        
        PDEType Type();
        
    private:
        PDEType _type;
    };
}
#endif //UNTITLED_ENERGYTRANSFEREQUATION_H
