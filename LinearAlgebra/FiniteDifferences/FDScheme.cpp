//
// Created by hal9000 on 4/19/23.
//


#include "FDScheme.h"

namespace LinearAlgebra{
            
        FDScheme::FDScheme(map<unsigned, FDSchemeComponent*>* components) {
            if (components->size() > 2) {
                throw invalid_argument("Only 1st and 2nd derivative schemes are supported.");
            }
            for (auto & component : *components) {
                auto derivativeOrder = component.first;
                if (derivativeOrder != 1 && derivativeOrder != 2) {
                    throw invalid_argument("Derivative order must be 1 and / or 2.");
                }
            }
            this->_components = components;
            
            schemeValuesAtAllPositions = calculateSchemeValuesAtAllPositions();
        }
        
        
        FDScheme::~FDScheme() {
            for (auto & component : *_components) {
                delete component.second;
            }
            delete _components;
            
            delete schemeValuesAtAllPositions;
        }

        FDSchemeComponent* FDScheme::getComponentOfDerivativeOrder(unsigned derivativeOrder) {
            return _components->at(derivativeOrder);
        }

    map<Position, double> *FDScheme::calculateSchemeValuesAtAllPositions() {
        for (auto & component : *_components) {
            auto derivativeOrder = component.first;
            auto schemeComponent = component.second;
            //schemeComponent.
        }
        return nullptr;
    }

}

