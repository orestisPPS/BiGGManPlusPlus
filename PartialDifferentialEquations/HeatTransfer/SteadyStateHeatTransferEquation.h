//
// Created by hal9000 on 12/9/22.
//

#ifndef UNTITLED_STEADYSTATEHEATTRANSFEREQUATION_H
#define UNTITLED_STEADYSTATEHEATTRANSFEREQUATION_H


#include <vector>

namespace PartialDifferentialEquations {

    class SteadyStateHeatTransferEquation {
    public:
        SteadyStateHeatTransferEquation(double thermalConductivity, std::vector<double> flowVelocity, double heatSource);
        ~SteadyStateHeatTransferEquation();
    };

    };

} // PartialDifferentialEquations

#endif //UNTITLED_STEADYSTATEHEATTRANSFEREQUATION_H
