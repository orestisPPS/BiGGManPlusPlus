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
        explicit FiniteDifferenceSchemeBuilder(shared_ptr<FDSchemeSpecs> schemeSpecs);

        // Map containing the sum of the weights for each position of the
        map<Position, double>* schemeWeightAtPosition;
        
        // The maximum number of points needed for any scheme in any direction.
        short unsigned getNumberOfGhostNodesNeeded() const;
        
        // Can be implemented for more efficient node graph creation.
        // Returns a map with all positions needed for the scheme and the number of points needed for each position.
        // Use this when creating a scheme that is consistent across the whole domain.
        map<Position, short unsigned> getNumberOfDiagonalNeighboursNeeded() const;
        
        unsigned getMaximumNumberOfPointsForArbitrarySchemeType() const;
        
        Scheme getSchemeAtDirection(Direction direction, unsigned derivativeOrder, unsigned errorOrder) const;
        
        shared_ptr<FDSchemeSpecs> _schemeSpecs;
        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a first derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        static map<unsigned, map<FDSchemeType, int>> schemeOrderToSchemeTypePointsDerivative1();

        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a first derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        static map<unsigned, map<FDSchemeType, int>> schemeOrderToSchemeTypePointsDerivative2();

        static map<Direction, map<FDSchemeType, vector<Position>>> schemeTypeToPositions();
        
        static map<Direction, map<vector<Position>, FDSchemeType>> positionsToSchemeType();
        
        //Use this when the error order is fixed and the scheme varies across the domain.
        static void templatePositionsAndPoints(short unsigned derivativeOrder, short unsigned errorOrder,
                                        vector<Direction>& directions,
                                        map<Direction, map<vector<Position>, short int>>& positionsAndPoints);

        static Scheme getSchemeWeightsFromQualifiedPositions(map<vector<Position>, short>& qualifiedPositionsAndPoints,
                                                              Direction& direction, unsigned short errorOrder, unsigned short derivativeOrder);

        map<vector<Position>,short> getQualifiedFromAvailable(map<vector<Position>,unsigned short>& availablePositionsAndPoints,
                                                               map<vector<Position>,short>& templatePositionsAndPoints);

        map<short unsigned, map<Direction, map<vector<Position>, short>>>
        initiatePositionsAndPointsMap(short unsigned& maxDerivativeOrder, vector<Direction>& directions);
    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEBUILDER_H
 