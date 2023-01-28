//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FINITEDIFFERENCESCHEMECOEFFICIENTCALCULATOR_H
#define UNTITLED_FINITEDIFFERENCESCHEMECOEFFICIENTCALCULATOR_H

#include "../../PositioningInSpace/DirectionsPositions.h"
#include "../../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "FDSchemeSpecs.h"
#include <map>
using namespace std;
using namespace PositioningInSpace;

namespace LinearAlgebra {

    class FiniteDifferenceSchemeCoefficientCalculator {
        
    public:
        FiniteDifferenceSchemeCoefficientCalculator(FDSchemeSpecs &schemeSpecs);
    private:
        static void correlateSpaceAndSchemeSpecs(FDSchemeSpecs &schemeSpecs);
        bool IsSpaceAndSchemeSpecsCorelated(FDSchemeSpecs &schemeSpecs);
    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMECOEFFICIENTCALCULATOR_H
