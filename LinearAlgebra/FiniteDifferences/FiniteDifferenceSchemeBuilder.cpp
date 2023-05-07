//
// Created by hal9000 on 4/7/23.
//

#include "FiniteDifferenceSchemeBuilder.h"

namespace LinearAlgebra {
    
    //Creates a Finite Difference Scheme For an input 
    FiniteDifferenceSchemeBuilder::FiniteDifferenceSchemeBuilder(FDSchemeSpecs* schemeSpecs) {
        this->_schemeSpecs = schemeSpecs;
    }
    
    short unsigned FiniteDifferenceSchemeBuilder::getNumberOfGhostNodesNeeded() {
        short unsigned max = 0;
        // March through all derivative orders
        for (auto &derivativeOrder: *this->_schemeSpecs->schemeTypeAndOrderAtDirectionForDerivativeOrder) {
            //March through all directions
            for (auto &direction: derivativeOrder.second) {
                //Scheme Type at direction
                auto schemeType = get<0>(direction.second);
                //Scheme Order at directionI
                auto order = get<1>(direction.second);

                // Get the points needed for the scheme at the current direction and the corresponding order from 
                // _schemeOrderToSchemeTypePointsDerivative1
                auto pointsNeededAtDirection = _schemeOrderToSchemeTypePointsDerivative1()[order][schemeType];
                if (pointsNeededAtDirection > max)
                    max = pointsNeededAtDirection;
            }
        }
        return max;
    }

    map<Position, short unsigned> FiniteDifferenceSchemeBuilder:: getNumberOfDiagonalNeighboursNeeded() {
        auto numberOfDiagonalNeighboursNeeded = map<Position, short unsigned>();
        //March through all derivative orders
        for (auto& derivativeOrder : *this->_schemeSpecs->schemeTypeAndOrderAtDirectionForDerivativeOrder) {
            //Derivative Order
            auto derivativeOrderI = derivativeOrder.first;

            //March through all directions
            for (auto &direction: derivativeOrder.second) {
                //Direction
                auto directionI = direction.first;

                //Scheme Type at directionI
                auto schemeTypeAtDirection = get<0>(direction.second);

                //Scheme Order at directionI
                auto orderAtDirection = get<1>(direction.second);

                // Get the points needed for the scheme at the current direction and the corresponding order from 
                // _schemeOrderToSchemeTypePointsDerivative1
                auto pointsNeededAtDirection =
                        _schemeOrderToSchemeTypePointsDerivative1()[orderAtDirection][schemeTypeAtDirection];

                //Get the positions for the scheme at the current direction
                auto positionsForSchemeAtDirection =
                        _schemeToPositions()[directionI][schemeTypeAtDirection];
                
                for (auto &position : positionsForSchemeAtDirection) {
                    numberOfDiagonalNeighboursNeeded[position] = pointsNeededAtDirection;
                }
            }
        }
        return numberOfDiagonalNeighboursNeeded;
    }
    
    vector<double> FiniteDifferenceSchemeBuilder::getSchemeWeightsAtDirection(Direction direction) {
        auto schemeTypeAndOrder = this->_schemeSpecs->
                schemeTypeAndOrderAtDirectionForDerivativeOrder->at(1)[direction];
        return FiniteDifferenceSchemeWeightsStructuredGrid::getWeightsVectorDerivative1(
                get<0>(schemeTypeAndOrder), get<1>(schemeTypeAndOrder));
    }
    
    

    map<unsigned int, map<FDSchemeType, int>>
    FiniteDifferenceSchemeBuilder::_schemeOrderToSchemeTypePointsDerivative1() {
        
        auto orderToPointsNeededPerDirection = map<unsigned int, map<FDSchemeType, int>>();
        //Cut-off error order O(Δx)
        orderToPointsNeededPerDirection[1] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[1][Forward] = 1;
        orderToPointsNeededPerDirection[1][Backward] = 1;
        orderToPointsNeededPerDirection[1][Central] = -1;
        
        //Cut-off error order O(Δx^2)
        orderToPointsNeededPerDirection[2] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[2][Forward] = 2;
        orderToPointsNeededPerDirection[2][Backward] = 2;
        orderToPointsNeededPerDirection[2][Central] = 2 / 2;
        
        //Cut-off error order O(Δx^3)
        orderToPointsNeededPerDirection[3] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[3][Forward] = 3;
        orderToPointsNeededPerDirection[3][Backward] = 3;
        orderToPointsNeededPerDirection[3][Central] = -1;
        
        //Cut-off error order O(Δx^4)
        orderToPointsNeededPerDirection[4] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[4][Forward] = 4;
        orderToPointsNeededPerDirection[4][Backward] = 4;
        orderToPointsNeededPerDirection[4][Central] = 6 / 2;
        
        //Cut-off error order O(Δx^5)
        orderToPointsNeededPerDirection[5] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[5][Forward] = 5;
        orderToPointsNeededPerDirection[5][Backward] = 5;
        orderToPointsNeededPerDirection[5][Central] = -1;
        
        return orderToPointsNeededPerDirection;
    }

    map<unsigned int, map<FDSchemeType, int>>
    FiniteDifferenceSchemeBuilder::_schemeOrderToSchemeTypePointsDerivative2() {

        auto orderToPointsNeededPerDirection = map<unsigned int, map<FDSchemeType, int>>();
        
        //Cut-off error order O(Δx^2)
        orderToPointsNeededPerDirection[2] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[2][Forward] = 3;
        orderToPointsNeededPerDirection[2][Backward] = 3;
        orderToPointsNeededPerDirection[2][Central] = 2 / 2;
        
        //Cut-off error order O(Δx^3)
        orderToPointsNeededPerDirection[3] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[3][Forward] = 4;
        orderToPointsNeededPerDirection[3][Backward] = 4;
        orderToPointsNeededPerDirection[3][Central] =-1;
        
        //Cut-off error order O(Δx^4)
        orderToPointsNeededPerDirection[4] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[4][Forward] = 4;
        orderToPointsNeededPerDirection[4][Backward] = 4;
        orderToPointsNeededPerDirection[4][Central] = 4  / 2;
        
        //Cut-off error order O(Δx^5)
        orderToPointsNeededPerDirection[5] = map<FDSchemeType, int>();
        orderToPointsNeededPerDirection[5][Forward] = 5;
        orderToPointsNeededPerDirection[5][Backward] = 5;
        orderToPointsNeededPerDirection[5][Central] = -1;
        
        return orderToPointsNeededPerDirection;
    }
    
    map<Direction, map<FDSchemeType, vector<Position>>>
    FiniteDifferenceSchemeBuilder::_schemeToPositions(){
        auto positionsForSchemeAtDirection = map<Direction, map<FDSchemeType, vector<Position>>>();
        //Direction 1
        auto dimensionOnePositions = map<FDSchemeType, vector<Position>> {
                {Central, {Left, Right}},
                {Forward, {Right}},
                {Backward,{Left}}
        };
        //Direction 2
        auto dimensionTwoPositions = map<FDSchemeType, vector<Position>> {
                {Central, {Bottom, Top}},
                {Forward, {Top}},
                {Backward, {Bottom}}
        };
        //Direction 3
        auto dimensionThreePositions = map<FDSchemeType, vector<Position>> {
                {Central, {Back, Front}},
                {Forward, {Front}},
                {Backward, {Back}}
        }; 
        positionsForSchemeAtDirection[One] = dimensionOnePositions;
        positionsForSchemeAtDirection[Two] = dimensionTwoPositions;
        positionsForSchemeAtDirection[Three] = dimensionThreePositions;
        return positionsForSchemeAtDirection;
    };

    map<Direction, map<vector<Position>, FDSchemeType>>
    FiniteDifferenceSchemeBuilder::_positionsToScheme(){
        auto positionsForSchemeAtDirection = map<Direction, map<vector<Position>, FDSchemeType>>();
        //Direction 1
        auto dimensionOnePositions = map<vector<Position>, FDSchemeType> {
                {{Left, Right}, Central},
                {{Right}, Forward},
                {{Left}, Backward}
        };
        //Direction 2
        auto dimensionTwoPositions = map<vector<Position>, FDSchemeType>{
                {{Bottom, Top}, Central},
                {{Top}, Forward},
                {{Bottom}, Backward}
        };
        //Direction 3
        auto dimensionThreePositions = map<vector<Position>, FDSchemeType>{
                {{Back, Front}, Central},
                {{Front}, Forward},
                {{Back}, Backward}
        };
        positionsForSchemeAtDirection[One] = dimensionOnePositions;
        positionsForSchemeAtDirection[Two] = dimensionTwoPositions;
        positionsForSchemeAtDirection[Three] = dimensionThreePositions;
        return positionsForSchemeAtDirection;
    }

    unsigned FiniteDifferenceSchemeBuilder::getMaximumNumberOfPointsForArbitrarySchemeType() {
        auto maxDerivativeOrder =
                static_cast<unsigned int>(_schemeSpecs->schemeTypeAndOrderAtDirectionForDerivativeOrder->size());
        auto maxOrder = -1;
        auto maxPoints = 0;
        for (auto &derivativeOrder :
             *_schemeSpecs->schemeTypeAndOrderAtDirectionForDerivativeOrder) {
            for (auto &direction: derivativeOrder.second) {
                auto errorOrder = get<1>(direction.second);
                if (errorOrder > maxOrder) {
                    maxOrder = errorOrder;
                }
            }
        }
        auto maxOrderScheme = map<FDSchemeType, int>();
        if (maxDerivativeOrder == 1){
            maxOrderScheme = _schemeOrderToSchemeTypePointsDerivative1()[maxOrder];
        }
        else if (maxDerivativeOrder == 2) {
            maxOrderScheme = _schemeOrderToSchemeTypePointsDerivative2()[maxOrder];
        }
        for (auto &schemeType : maxOrderScheme) {
            if (schemeType.second > maxPoints) {
                maxPoints = schemeType.second;
            }
        }
        return maxPoints;
    };
} // LinearAlgebra


/*
map<FDSchemeType, vector<Position>> FiniteDifferenceSchemeBuilder::_schemeTypeToPositionsOfPointsNeeded() {
    map<FDSchemeType, vector<Position>> schemeTypeToPositionsOfPointsNeeded;

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
}*/
