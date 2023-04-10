//
// Created by hal9000 on 4/7/23.
//

#include "FiniteDifferenceSchemeBuilder.h"

namespace LinearAlgebra {
    
    FiniteDifferenceSchemeBuilder::FiniteDifferenceSchemeBuilder(FDSchemeSpecs* schemeSpecs) {
        this->_schemeSpecs = schemeSpecs;
        
    }

    map<unsigned int, map<FiniteDifferenceSchemeType, int>>
    FiniteDifferenceSchemeBuilder::_schemeOrderToSchemeTypePointsNeeded() {
        
        auto orderToPointsNeeded = map<unsigned int, map<FiniteDifferenceSchemeType, int>>();
        //Cut-off error order O(Δx)
        orderToPointsNeeded[1] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeeded[1][Forward] = 1;
        orderToPointsNeeded[1][Backward] = 1;
        orderToPointsNeeded[1][Central] = -1;
        
        //Cut-off error order O(Δx^2)
        orderToPointsNeeded[2] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeeded[2][Forward] = 2;
        orderToPointsNeeded[2][Backward] = 2;
        orderToPointsNeeded[2][Central] = 2;
        
        //Cut-off error order O(Δx^3)
        orderToPointsNeeded[3] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeeded[3][Forward] = 3;
        orderToPointsNeeded[3][Backward] = 3;
        orderToPointsNeeded[3][Central] = 4;
        
        //Cut-off error order O(Δx^4)
        orderToPointsNeeded[4] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeeded[4][Forward] = 4;
        orderToPointsNeeded[4][Backward] = 4;
        orderToPointsNeeded[4][Central] = 6;
        
        //Cut-off error order O(Δx^5)
        orderToPointsNeeded[5] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeeded[5][Forward] = 5;
        orderToPointsNeeded[5][Backward] = 5;
        orderToPointsNeeded[5][Central] = 8;
        
        return orderToPointsNeeded;
    }

     /**
     * Builds a map from FiniteDifferenceSchemeType to a list of Position enum values
     * specifying the relative positions of the neighboring nodes needed for finite
     * difference schemes.
     */
     map<FiniteDifferenceSchemeType, list<Position>> FiniteDifferenceSchemeBuilder::_schemeTypeToPositionsOfPointsNeeded() {
         map<FiniteDifferenceSchemeType, list<Position>> schemeTypeToPositionsOfPointsNeeded;

         // Forward scheme
         list<Position> forwardPositions = {
                 TopLeft, Top, TopRight, Left, Center, Right, FrontTopLeft, FrontTop, FrontTopRight,
                 FrontLeft, Front, FrontRight, BackTopLeft, BackTop, BackTopRight, BackLeft, Back, BackRight
         };
         schemeTypeToPositionsOfPointsNeeded[Forward] = forwardPositions;

         // Backward scheme
         list<Position> backwardPositions = {
                 BottomLeft, Bottom, BottomRight, Left, Center, Right, FrontBottomLeft, FrontBottom, FrontBottomRight,
                 FrontLeft, Front, FrontRight, BackBottomLeft, BackBottom, BackBottomRight, BackLeft, Back, BackRight
         };
         schemeTypeToPositionsOfPointsNeeded[Backward] = backwardPositions;

         // Central scheme
         list<Position> centralPositions = {
                 TopLeft, Top, TopRight, Left, Center, Right, BottomLeft, Bottom, BottomRight,
                 FrontTopLeft, FrontTop, FrontTopRight, FrontLeft, Front, FrontRight,
                 FrontBottomLeft, FrontBottom, FrontBottomRight, BackTopLeft, BackTop, BackTopRight,
                 BackLeft, Back, BackRight, BackBottomLeft, BackBottom, BackBottomRight
         };
         schemeTypeToPositionsOfPointsNeeded[Central] = centralPositions;

         return schemeTypeToPositionsOfPointsNeeded;
     }
} // LinearAlgebra