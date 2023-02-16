//
// Created by hal9000 on 1/28/23.
//

#include <stdexcept>
#include "FiniteDifferenceSchemeCoefficientCalculator.h"

namespace LinearAlgebra {
    
    FiniteDifferenceSchemeCoefficientCalculator::FiniteDifferenceSchemeCoefficientCalculator(FDSchemeSpecs &schemeSpecs) {
        correlateSpaceAndSchemeSpecs(schemeSpecs);
    }
    
    void FiniteDifferenceSchemeCoefficientCalculator::correlateSpaceAndSchemeSpecs(FDSchemeSpecs &schemeSpecs) {

    }
    
    map<Position,int> FiniteDifferenceSchemeCoefficientCalculator::NormalNeighboursSigns(){
        map<Position,int> neighbourSigns = { {Top, 1}, {Bottom, -1},
                                             {Left, -1}, {Right, 1},
                                             {Front, 1}, {Back, -1}};
        return neighbourSigns;    
    }
    
    map<Position,tuple<int,int>> FiniteDifferenceSchemeCoefficientCalculator::DiagonalNeigboursSigns(){
        map<Position,tuple<int,int>> neighbourSigns = { {TopLeft, {-1,1}}, {TopRight, {1,1}},
                                                       {BottomLeft, {-1,-1}}, {BottomRight, {1,-1}},
                                                       {FrontTopLeft, {-1,1}}, {FrontTopRight, {1,1}},
                                                       {FrontBottomLeft, {-1,-1}}, {FrontBottomRight, {1,-1}},
                                                       {BackTopLeft, {-1,1}}, {BackTopRight, {1,1}},
                                                       {BackBottomLeft, {-1,-1}}, {BackBottomRight, {1,-1}}};
        return neighbourSigns;
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