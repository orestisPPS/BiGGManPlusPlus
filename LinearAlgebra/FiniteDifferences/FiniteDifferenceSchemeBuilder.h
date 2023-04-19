//
// Created by hal9000 on 4/7/23.
//

#ifndef UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H
#define UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H

#include "FDSchemeSpecs.h"
using namespace LinearAlgebra;

namespace LinearAlgebra {

    class FiniteDifferenceSchemeBuilder {
    public:
        FiniteDifferenceSchemeBuilder(FDSchemeSpecs* schemeSpecs);

        // Map containing the sum of the weights for each position of the
        map<Position, double>* schemeWeightAtPosition;
        
        
    private:
        
        FDSchemeSpecs* _schemeSpecs;
        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        static map<unsigned, map<FDSchemeType, int>> _schemeOrderToSchemeTypePointsNeeded();
        
        //Maps the type of the scheme to the appropriate relative positions of the points needed to build it
        //For example, type Forward, (Right, Front, Top)
        static map<FDSchemeType, vector<Position>> _schemeTypeToPositionsOfPointsNeeded();


        static map<unsigned int, map<FDSchemeType, vector<Position>>> positionsForFDSchemeTypeAndDimensions();

        static map<Direction, map<FDSchemeType, vector<Position>>> positionsForSchemeAtDirection();
    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEBUILDER_H
 