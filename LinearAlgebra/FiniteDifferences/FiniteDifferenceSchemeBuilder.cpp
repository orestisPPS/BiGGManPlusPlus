//
// Created by hal9000 on 4/7/23.
//

#include "FiniteDifferenceSchemeBuilder.h"

namespace LinearAlgebra {
    
    FiniteDifferenceSchemeBuilder::FiniteDifferenceSchemeBuilder(FDSchemeSpecs* schemeSpecs) {
        this->_schemeSpecs = schemeSpecs;
        auto dimensions = this->_schemeSpecs->schemeTypeAndOrderAtDirection.size();
        //Contains all the points NEEDED for each input Finite Difference Scheme type at all directions.
        map<FiniteDifferenceSchemeType,int> schemeTypeToPointsNeeded = map<FiniteDifferenceSchemeType,int>();
        
        //Search all the directions of the scheme.
        for (auto& direction : this->_schemeSpecs->schemeTypeAndOrderAtDirection) {
            //Get the type and order of the scheme at the current direction
            auto schemeTypeAtDirection = get<0>(direction.second);
            auto orderAtDirection = get<1>(direction.second);
            // Get the points needed for the scheme at the current direction and the corresponding order from 
            // _schemeOrderToSchemeTypePointsNeeded
            auto pointsNeededAtDirection = _schemeOrderToSchemeTypePointsNeeded()[orderAtDirection][schemeTypeAtDirection];
            schemeTypeToPointsNeeded[schemeTypeAtDirection] = pointsNeededAtDirection;
        }
        
        //Create a template map that goes as 
    }

    map<unsigned int, map<FiniteDifferenceSchemeType, int>>
    FiniteDifferenceSchemeBuilder::_schemeOrderToSchemeTypePointsNeeded() {
        
        auto orderToPointsNeededPerDirection = map<unsigned int, map<FiniteDifferenceSchemeType, int>>();
        //Cut-off error order O(Δx)
        orderToPointsNeededPerDirection[1] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeededPerDirection[1][Forward] = 1;
        orderToPointsNeededPerDirection[1][Backward] = 1;
        orderToPointsNeededPerDirection[1][Central] = -1;
        
        //Cut-off error order O(Δx^2)
        orderToPointsNeededPerDirection[2] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeededPerDirection[2][Forward] = 2;
        orderToPointsNeededPerDirection[2][Backward] = 2;
        orderToPointsNeededPerDirection[2][Central] = 2 / 2;
        
        //Cut-off error order O(Δx^3)
        orderToPointsNeededPerDirection[3] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeededPerDirection[3][Forward] = 3;
        orderToPointsNeededPerDirection[3][Backward] = -1;
        
        //Cut-off error order O(Δx^4)
        orderToPointsNeededPerDirection[4] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeededPerDirection[4][Forward] = 4;
        orderToPointsNeededPerDirection[4][Backward] = 4;
        orderToPointsNeededPerDirection[4][Central] = 6 / 2;
        
        //Cut-off error order O(Δx^5)
        orderToPointsNeededPerDirection[5] = map<FiniteDifferenceSchemeType, int>();
        orderToPointsNeededPerDirection[5][Forward] = 5;
        orderToPointsNeededPerDirection[5][Backward] = 5;
        orderToPointsNeededPerDirection[5][Central] = -1;
        
        return orderToPointsNeededPerDirection;
    }

     /**
     * Builds a map from FiniteDifferenceSchemeType to a list of Position enum values
     * specifying the relative positions of the neighboring nodes needed for finite
     * difference schemes.
     */
     map<FiniteDifferenceSchemeType, vector<Position>> FiniteDifferenceSchemeBuilder::_schemeTypeToPositionsOfPointsNeeded() {
         map<FiniteDifferenceSchemeType, vector<Position>> schemeTypeToPositionsOfPointsNeeded;

         // Forward scheme
         vector<Position> forwardPositions = { Right, Front, Top };
         schemeTypeToPositionsOfPointsNeeded[Forward] = forwardPositions;

         // Backward scheme
         vector<Position> backwardPositions = { Left, Back, Bottom };
         schemeTypeToPositionsOfPointsNeeded[Backward] = backwardPositions;

         // Central scheme
         vector<Position> centralPositions = {
            Right, Left, Top, Bottom, Front, Back
         };
         schemeTypeToPositionsOfPointsNeeded[Central] = centralPositions;
         
         vector<Position> diagonalPositions = {
            LeftTopBack, TopBack, RightTopBack,
            LeftTop, RightTop,
            LeftTopFront, TopFront, RightTopFront,
            LeftBack, RightBack,
            LeftFront, RightFront,
            LeftBottomBack, BottomBack, RightBottomBack,
            LeftBottom, RightBottom,
            LeftBottomFront, BottomFront, RightBottomFront
         };
        schemeTypeToPositionsOfPointsNeeded[Mixed] = diagonalPositions;
        return schemeTypeToPositionsOfPointsNeeded;
     }




    map<Direction, map<FiniteDifferenceSchemeType, vector<Position>>>
    FiniteDifferenceSchemeBuilder::positionsForSchemeAtDirection(){
        auto positionsForSchemeAtDirection = map<Direction, map<FiniteDifferenceSchemeType, vector<Position>>>();
        //Direction 1
        auto dimensionOnePositions = map<FiniteDifferenceSchemeType, vector<Position>> {
                {Central, {Right, Left}},
                {Forward, {Right}},
                {Backward, {Left}}
        };
        //Direction 2
        auto dimensionTwoPositions = map<FiniteDifferenceSchemeType, vector<Position>> {
                {Central, {Top, Bottom}},
                {Forward, {Top}},
                {Backward, {Bottom}}
        };
        //Direction 3
        auto dimensionThreePositions = map<FiniteDifferenceSchemeType, vector<Position>> {
                {Central, {Front, Back}},
                {Forward, {Front}},
                {Backward, {Back}}
        }; 
        positionsForSchemeAtDirection[One] = dimensionOnePositions;
        positionsForSchemeAtDirection[Two] = dimensionTwoPositions;
        positionsForSchemeAtDirection[Three] = dimensionThreePositions;
        return positionsForSchemeAtDirection;
    };
} // LinearAlgebra