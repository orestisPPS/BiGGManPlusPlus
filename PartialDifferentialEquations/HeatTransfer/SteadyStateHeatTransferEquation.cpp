//
// Created by hal9000 on 12/9/22.
//

#include "EnergyTransferEquation.h"

namespace PartialDifferentialEquations {
    
        EnergyTransferEquation::EnergyTransferEquation(double *thermalConductivity, std::vector<double> *flowVelocity, double *heatSource) {
            auto secondDerivativeProperties = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (i == j && i < 3) {
                        secondDerivativeProperties->populateElement(i, j, *thermalConductivity);
                    } else {
                        secondDerivativeProperties->populateElement(i, j, 0);
                    }
                }
            }
            
            auto firstDerivativeProperties = new vector<double>;
            for (int i = 0; i < 3; ++i) {
                firstDerivativeProperties->push_back((*flowVelocity)[i]);
            }
            firstDerivativeProperties->push_back(0);
            
            auto zerothDerivativeProperties = new double(0);
            
            auto isTransient = new bool(false);
            
            properties = new SecondOrderLinearPDEProperties(secondDerivativeProperties,
                                                            firstDerivativeProperties,
                                                            zerothDerivativeProperties,
                                                            heatSource,
                                                            isTransient);
            

        }
    
        EnergyTransferEquation::EnergyTransferEquation(Primitives::Matrix<double> anisotropicThermalConductivity, std::vector<double> flowVelocity, double heatSource) {
            auto secondDerivativeProperties = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (i == j && i < 3) {
                        secondDerivativeProperties->populateElement(i, j, anisotropicThermalConductivity.element(i, j));
                    } else {
                        secondDerivativeProperties->populateElement(i, j, 0);
                    }
                }
            }
            
            auto firstDerivativeProperties = new vector<double>;
            for (int i = 0; i < 4; ++i) {
                if (i < 3) {
                    firstDerivativeProperties->push_back(flowVelocity[i]);
                } else {
                    firstDerivativeProperties->push_back(0);
                }
            }
            
            auto zerothDerivativeProperties = new double(0);
            
            auto source = new double(heatSource);
            
            auto isTransient = new bool(false);
            
            properties = new SecondOrderLinearPDEProperties(secondDerivativeProperties,
                                                            firstDerivativeProperties,
                                                            zerothDerivativeProperties,
                                                            source,
                                                            isTransient);
        }
    
        EnergyTransferEquation::EnergyTransferEquation(
                map<int *, double> locallyAnisotropicThermalConductivity, map<int *, double> localFlowVelocity,
                map<int *, double> heatSource) {
            auto secondDerivativeProperties = new map<int*, double>;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    if (i == j && i < 3) {
                        secondDerivativeProperties->insert(pair<int*, double>(new int[2]{i, j}, locallyAnisotropicThermalConductivity.at(new int[2]{i, j})));
                    } else {
                        secondDerivativeProperties->insert(pair<int*, double>(new int[2]{i, j}, 0));
                    }
                }
            }
        }
        
        EnergyTransferEquation::~EnergyTransferEquation() = default;
    
    } // PartialDifferentialEquations
} // PartialDifferentialEquations