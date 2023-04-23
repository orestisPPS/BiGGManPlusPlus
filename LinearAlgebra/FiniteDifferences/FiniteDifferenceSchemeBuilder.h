//
// Created by hal9000 on 4/7/23.
//

#ifndef UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H
#define UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H

#include "FDSchemeSpecs.h"
#include "FirstOrderDerivativeFDSchemeCalculator.h"
#include "FiniteDifferenceSchemeWeightsCalculator.h"
#include "../../Discretization/Node/IsoparametricNodeGraph.h"
using namespace Discretization;

using namespace LinearAlgebra;

namespace LinearAlgebra {

    class FiniteDifferenceSchemeBuilder {
    public:
        //
        FiniteDifferenceSchemeBuilder(FDSchemeSpecs* schemeSpecs);

        // Map containing the sum of the weights for each position of the
        map<Position, double>* schemeWeightAtPosition;
        
        map<Position, double>* scheme();
        
        // Creates A Finite Difference Scheme that is consistent across the whole domain.
        // For example, (Central, 2) will impose a central difference scheme of order 2 all directions and nodes/DOFs. 
        // In most cases a GhostPseudoMesh is needed. 
        void createConsistentScheme();
        
        // The maximum number of points needed for any scheme in any direction.
        short unsigned getNumberOfGhostNodesNeeded();
        
        // Can be implemented for more efficient node graph creation.
        // Returns a map with all positions needed for the scheme and the number of points needed for each position.
        // Use this when creating a scheme that is consistent across the whole domain.
        map<Position, short unsigned> getNumberOfDiagonalNeighboursNeeded();
        
        double calculateDerivative(IsoParametricNodeGraph* nodeGraph,  unsigned derivativeOrder, Direction direction);
        
        map<Direction, double> calculateDerivativeVector(IsoParametricNodeGraph* nodeGraph, unsigned derivativeOrder);
        
        vector<double> getSchemeWeightsAtDirection(Direction direction);
        
        
    private:
        
        FDSchemeSpecs* _schemeSpecs;
        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a first derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        static map<unsigned, map<FDSchemeType, int>> _schemeOrderToSchemeTypePointsNeededFirstDerivative();
        

        //Maps the order of the scheme to the type of the scheme and the neighbouring points needed to build it
        // for a second derivative finite difference scheme.
        //For example Order 2, type central, 3 points
        //If the second item of the tuple is -1, then the scheme is not defined for that order
        //For example Order 1, type central, -1 points
        static map<unsigned, map<FDSchemeType, int>> _schemeOrderToSchemeTypePointsNeededSeconddDerivative();
        
        static map<Direction, map<FDSchemeType, vector<Position>>> _positionsForSchemeAtDirection();
    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEBUILDER_H
 