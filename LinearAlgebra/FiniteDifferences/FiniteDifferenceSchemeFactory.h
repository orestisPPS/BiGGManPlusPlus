//
// Created by hal9000 on 4/7/23.
//

#ifndef UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H
#define UNTITLED_FIBITEDIFFERENCESCHEMEBUILDER_H

#include "FDSchemeSpecs.h"
#include "../../Discretization/Node/IsoparametricNodeGraph.h"
using namespace Discretization;

using namespace LinearAlgebra;

namespace LinearAlgebra {

    class FiniteDifferenceSchemeFactory {
    public:
        //
        FiniteDifferenceSchemeFactory(FDSchemeSpecs* schemeSpecs, IsoParametricNodeGraph* nodeGraph);

        // Map containing the sum of the weights for each position of the
        map<Position, double>* schemeWeightAtPosition;
        
        map<Position, double>* scheme();
        
        // Creates A Finite Difference Scheme that is consistent across the whole domain.
        // For example, (Central, 2) will impose a central difference scheme of order 2 all directions and nodes/DOFs. 
        // In most cases a GhostPseudoMesh is needed. 
        void createConsistentScheme();
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
 