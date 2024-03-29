//
// Created by hal9000 on 4/7/23.
//

#include "FiniteDifferenceSchemeBuilder.h"

namespace LinearAlgebra {
    
    //Creates a Finite Difference Scheme For an input 
    FiniteDifferenceSchemeBuilder::FiniteDifferenceSchemeBuilder(shared_ptr<FDSchemeSpecs> schemeSpecs) : _schemeSpecs(schemeSpecs) {
    }
    
    short unsigned FiniteDifferenceSchemeBuilder::getNumberOfGhostNodesNeeded() const {
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
                // schemeOrderToSchemeTypePointsDerivative1
                auto pointsNeededAtDirection = schemeOrderToSchemeTypePointsDerivative1()[order][schemeType];
                if (pointsNeededAtDirection > max)
                    max = pointsNeededAtDirection;
            }
        }
        return max;
    }

    map<Position, short unsigned> FiniteDifferenceSchemeBuilder:: getNumberOfDiagonalNeighboursNeeded() const {
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
                // schemeOrderToSchemeTypePointsDerivative1
                auto pointsNeededAtDirection =
                        schemeOrderToSchemeTypePointsDerivative1()[orderAtDirection][schemeTypeAtDirection];

                //Get the positions for the scheme at the current direction
                auto positionsForSchemeAtDirection =
                        schemeTypeToPositions()[directionI][schemeTypeAtDirection];
                
                for (auto &position : positionsForSchemeAtDirection) {
                    numberOfDiagonalNeighboursNeeded[position] = pointsNeededAtDirection;
                }
            }
        }
        return numberOfDiagonalNeighboursNeeded;
    }
    
    Scheme FiniteDifferenceSchemeBuilder::getSchemeAtDirection(Direction direction, unsigned derivativeOrder, unsigned errorOrder) const {
        auto schemeTypeAndOrder = this->_schemeSpecs->
                schemeTypeAndOrderAtDirectionForDerivativeOrder->at(derivativeOrder)[direction];
        return FiniteDifferenceSchemeWeightsStructuredGrid::getScheme(
                get<0>(schemeTypeAndOrder), derivativeOrder, errorOrder);
    }

    map<unsigned int, map<FDSchemeType, int>>
    FiniteDifferenceSchemeBuilder::schemeOrderToSchemeTypePointsDerivative1() {
        
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
    FiniteDifferenceSchemeBuilder::schemeOrderToSchemeTypePointsDerivative2() {

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
    FiniteDifferenceSchemeBuilder::schemeTypeToPositions(){
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
    FiniteDifferenceSchemeBuilder::positionsToSchemeType(){
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

    unsigned FiniteDifferenceSchemeBuilder::getMaximumNumberOfPointsForArbitrarySchemeType() const {
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
            maxOrderScheme = schemeOrderToSchemeTypePointsDerivative1()[maxOrder];
        }
        else if (maxDerivativeOrder == 2) {
            maxOrderScheme = schemeOrderToSchemeTypePointsDerivative2()[maxOrder];
        }
        for (auto &schemeType : maxOrderScheme) {
            if (schemeType.second > maxPoints) {
                maxPoints = schemeType.second;
            }
        }
        return maxPoints;
    }

    void FiniteDifferenceSchemeBuilder::templatePositionsAndPoints
    (short unsigned derivativeOrder, short unsigned errorOrder, vector<Direction>& directions,
     map<Direction, map<vector<Position>, short int>>& templatePositionsAndPoints) {
        //Find the number of points needed for the desired order of accuracy.
        //The scheme type varies depending on the available neighbours of the dof.
        auto schemeTypeToPoints = map<FDSchemeType, int>();
        switch (derivativeOrder) {
            case 1:
                schemeTypeToPoints = schemeOrderToSchemeTypePointsDerivative1()[errorOrder];
                break;
            case 2:
                schemeTypeToPoints = schemeOrderToSchemeTypePointsDerivative2()[errorOrder];
                break;
            default:
                throw invalid_argument("Derivative order must be 1 or 2");
        }
        //Convert scheme type to positions
        auto schemeTypeToPosition = schemeTypeToPositions();
        for (auto direction : directions) {
            auto dir = static_cast<Direction>(direction);
            templatePositionsAndPoints.insert(pair<Direction, map<vector<Position>, short int>> (
                    dir, map<vector<Position>, short int>()));

            auto schemeTypeToPositions = schemeTypeToPosition[dir];
            for (auto &schemeTypePositionTuple : schemeTypeToPositions) {
                templatePositionsAndPoints[dir].insert(pair<vector<Position>, short int>(
                        schemeTypePositionTuple.second, schemeTypeToPoints[schemeTypePositionTuple.first]));
            }
        }
        //sort
        for (auto &direction : templatePositionsAndPoints) {
            auto &positionsAndPoints = direction.second;
            for (auto &positionAndPoint : positionsAndPoints) {
                auto positionVector = positionAndPoint.first;
                sort(positionVector.begin(), positionVector.end(),
                     [](Position a, Position b) { return a < b; });
            }
        }
    }

    Scheme FiniteDifferenceSchemeBuilder::getSchemeWeightsFromQualifiedPositions(
            map<vector<Position>, short>& qualifiedPositionsAndPoints,
            Direction& direction,
            unsigned short errorOrder,
            unsigned short derivativeOrder) {
        map<FDSchemeType, Scheme> availableSchemes;
        auto positionsToScheme = positionsToSchemeType();

        for (auto& qualifiedPositionAndPoint : qualifiedPositionsAndPoints) {
            const vector<Position>& positionVector = qualifiedPositionAndPoint.first;
            FDSchemeType schemeType = positionsToScheme[direction][positionVector];
            availableSchemes[schemeType] = FiniteDifferenceSchemeWeightsStructuredGrid::getScheme(schemeType, derivativeOrder, errorOrder);
        }
        
        vector<Position> positionsFromScheme;
        Scheme weights;

        if (availableSchemes.count(Central) != 0) {
            positionsFromScheme = schemeTypeToPositions()[direction][Central];
            weights = availableSchemes[Central];
        }
        else if (availableSchemes.count(Forward) != 0) {
            positionsFromScheme = schemeTypeToPositions()[direction][Forward];
            return availableSchemes[Forward];
        }
        else if (availableSchemes.count(Backward) != 0) {
            positionsFromScheme = schemeTypeToPositions()[direction][Backward];
            return availableSchemes[Backward];
        }
        else {
            throw invalid_argument("No scheme found for the given positions");
        }
        for (auto it = qualifiedPositionsAndPoints.begin(); it != qualifiedPositionsAndPoints.end();) {
            if (it->first != positionsFromScheme) {
                it = qualifiedPositionsAndPoints.erase(it);
            }
            else {
                ++it;
            }
        }
        return weights;
    }


    map<vector<Position>,short> FiniteDifferenceSchemeBuilder::
    getQualifiedFromAvailable(map<vector<Position>,unsigned short>& availablePositionsAndPoints,
                               map<vector<Position>,short>& templatePositionsAndPoints){
        map<vector<Position>,short> qualifiedPositionsAndPoints = map<vector<Position>,short>();
        //Check if the specifications of the template positions and points are met in the available positions and points
        for (auto &templatePositionAndPoints : templatePositionsAndPoints) {
            for (auto &availablePositionAndPoints : availablePositionsAndPoints) {
                //Check if the template positions and points are met in the available positions and points
                if (availablePositionAndPoints.first == templatePositionAndPoints.first &&
                    availablePositionAndPoints.second >= templatePositionAndPoints.second) {
                    qualifiedPositionsAndPoints.insert(pair<vector<Position>, short>(
                            templatePositionAndPoints.first, templatePositionAndPoints.second));
                }
            }
        }
        return qualifiedPositionsAndPoints;
    }

    map<short unsigned, map<Direction, map<vector<Position>, short>>> FiniteDifferenceSchemeBuilder::
    initiatePositionsAndPointsMap(short unsigned& maxDerivativeOrder, vector<Direction>& directions) {
        map<short unsigned, map<Direction, map<vector<Position>, short>>> positionsAndPoints;
        for (short unsigned derivativeOrder = 1; derivativeOrder <= maxDerivativeOrder; derivativeOrder++) {
            positionsAndPoints.insert(pair<short unsigned, map<Direction, map<vector<Position>, short>>>(
                    derivativeOrder, map<Direction, map<vector<Position>, short>>()));
            for (auto &direction : directions) {
                positionsAndPoints[derivativeOrder].insert(pair<Direction, map<vector<Position>, short>>(
                        direction, map<vector<Position>, short>()));
            }
        }
        return positionsAndPoints;
    }
    
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
