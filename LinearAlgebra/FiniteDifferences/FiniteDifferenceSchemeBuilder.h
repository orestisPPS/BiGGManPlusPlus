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
        
        vector<double> getSchemeWeightsAtDirection(Direction direction);
        
        
    private:
        
        FDSchemeSpecs* _schemeSpecs;
        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a first derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        static map<unsigned, map<FDSchemeType, int>> _schemeOrderToSchemeTypePointsDerivative1();

        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a first derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        static map<unsigned, map<FDSchemeType, int>> _schemeOrderToSchemeTypePointsDerivative2();

        
        static map<Direction, map<FDSchemeType, vector<Position>>> _schemeToPositions();
        
        static map<Direction, map<vector<Position>, FDSchemeType>> _positionsToScheme();
    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEBUILDER_H
 