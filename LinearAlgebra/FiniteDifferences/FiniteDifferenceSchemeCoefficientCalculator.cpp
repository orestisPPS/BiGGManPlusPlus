//
// Created by hal9000 on 1/28/23.
//

#include <stdexcept>
#include "FiniteDifferenceSchemeCoefficientCalculator.h"

namespace LinearAlgebra {
    
    FiniteDifferenceSchemeCoefficientCalculator::FiniteDifferenceSchemeCoefficientCalculator(FDSchemeSpecs &schemeSpecs) {
        correlateSpaceAndSchemeSpecs();
    }
    
    void FiniteDifferenceSchemeCoefficientCalculator::correlateSpaceAndSchemeSpecs(FDSchemeSpecs &schemeSpecs) {

    }
    
    bool FiniteDifferenceSchemeCoefficientCalculator::IsSpaceAndSchemeSpecsCorelated(FDSchemeSpecs &schemeSpecs) {
        auto axis = schemeSpecs.space.directions();
        auto schemeSpecsDirections = schemeSpecs.schemeTypeAndOrderAtDirection;

        if (schemeSpecsDirections.count(Time) == 0 && schemeSpecs.schemeTypeAndOrderAtDirection.size() != axis.size() )
            throw std::invalid_argument("The number of directions in the scheme is not equal to the number of directions in the space");

        for (auto &direction : axis) {
            if (schemeSpecsDirections.count(direction) == 0)
                throw std::invalid_argument("Direction " + to_string(direction) + " is not defined in the scheme");
        }
    }
} // LinearAlgebra