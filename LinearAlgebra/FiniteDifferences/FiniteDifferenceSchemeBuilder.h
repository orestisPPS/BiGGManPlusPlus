//
// Created by hal9000 on 4/7/23.
//

#ifndef UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H
#define UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H

#include "FDSchemeSpecs.h"
#include "FiniteDifferenceSchemeWeightsStructuredGrid.h"
#include "../../Discretization/Node/IsoparametricNodeGraph.h"
using namespace Discretization;

using namespace LinearAlgebra;

namespace LinearAlgebra {

    class FiniteDifferenceSchemeBuilder {
    public:
        //
        explicit FiniteDifferenceSchemeBuilder(FDSchemeSpecs* schemeSpecs);

        // Map containing the sum of the weights for each position of the
        map<Position, double>* schemeWeightAtPosition;
        
        map<Position, double>* scheme();
        
        
        // The maximum number of points needed for any scheme in any direction.
        short unsigned getNumberOfGhostNodesNeeded();
        
        // Can be implemented for more efficient node graph creation.
        // Returns a map with all positions needed for the scheme and the number of points needed for each position.
        // Use this when creating a scheme that is consistent across the whole domain.
        map<Position, short unsigned> getNumberOfDiagonalNeighboursNeeded();
        
        unsigned getMaximumNumberOfPointsForArbitrarySchemeType();
        
        vector<double> getSchemeWeightsAtDirectionDerivative1(Direction direction);
        
        vector<double> getSchemeWeightsAtDirectionDerivative2(Direction direction);
        
        map<FDSchemeType, int> getSchemeTypeAndOrder(unsigned derivativeOrder,
                                                            unsigned int errorOrder);
        
        FDSchemeSpecs* _schemeSpecs;
        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a first derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        map<unsigned, map<FDSchemeType, int>> schemeOrderToSchemeTypePointsDerivative1();

        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a first derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        map<unsigned, map<FDSchemeType, int>> schemeOrderToSchemeTypePointsDerivative2();

        static map<Direction, map<FDSchemeType, vector<Position>>> schemeTypeToPositions();
        
        static map<Direction, map<vector<Position>, FDSchemeType>> positionsToSchemeType();
        
        //Use this when the error order is fixed and the scheme varies across the domain.
        void templatePositionsAndPoints(short unsigned derivativeOrder, short unsigned errorOrder,
                                        vector<Direction>& directions,
                                        map<Direction, map<vector<Position>, short int>>& positionsAndPoints);

        static vector<double> getSchemeWeightsFromQualifiedPositions(map<vector<Position>, short>& qualifiedPositionsAndPoints,
                                                              Direction& direction, unsigned short errorOrder, unsigned short derivativeOrder);
        
        static map<Direction, map<Position, unsigned short>>
        getNumberOfPointsForPositionsFromQualifiedPositions(map<vector<Position>, short>& qualifiedPositionsAndPoints);
        
    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEBUILDER_H
 