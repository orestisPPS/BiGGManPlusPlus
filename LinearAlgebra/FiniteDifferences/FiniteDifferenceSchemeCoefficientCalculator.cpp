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
        for (auto spaceDirection : schemeSpecs.space.directions()) {
            if (schemeSpecs.schemeTypeAndOrderAtDirection.find(spaceDirection) == schemeSpecs.schemeTypeAndOrderAtDirection.end()) {

            }
        }
    }

    list<list<Position>> FiniteDifferenceSchemeCoefficientCalculator::get1DPositionsAtDirection(Direction direction) {
        list<list<Position>> positions = {{Left}, {Right}};
        return positions;
    }

    list<list<Position>> FiniteDifferenceSchemeCoefficientCalculator::get2DPositionsAtDirection(Direction direction) {
        list <Position> horizontalNeighbours = {Left, Right};
        list <Position> verticalNeighbours = {Top, Bottom};
        list <Position> diagonalNeighbours = {TopLeft, TopRight, BottomLeft, BottomRight};
        list <list<Position>> positions = {horizontalNeighbours, verticalNeighbours, diagonalNeighbours};
        return positions;
    }

    list<list<Position>> FiniteDifferenceSchemeCoefficientCalculator::get3DPositionsAtDirection(Direction direction) {
        
        list <Position> horizontalNeighbours = {Left, Right};
        list <Position> verticalNeighbours = {Top, Bottom};
        list <Position> depthNeighbours = {Front, Back};
        list <Position> diagonalNeighbours = {TopLeft, TopRight, BottomLeft, BottomRight,
                                                            FrontTopLeft, FrontTopRight, FrontLeft, FrontRight, FrontBottomLeft, FrontBottomRight,
                                                            BackTopLeft, BackTopRight, BackLeft, BackRight, BackBottomLeft, BackBottomRight};
        auto positions = {horizontalNeighbours, verticalNeighbours, depthNeighbours, diagonalNeighbours};
        return positions;
    }
    
} // LinearAlgebra