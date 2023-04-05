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
        static list<list<Position>> get1DPositionsAtDirection(Direction direction);
        static list<list<Position>> get2DPositionsAtDirection(Direction direction);
        static list<list<Position>> get3DPositionsAtDirection(Direction direction);
        map<Position,int> NormalNeighboursSigns();
        map<Position,tuple<int,int>> DiagonalNeigboursSigns();
    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMECOEFFICIENTCALCULATOR_H
