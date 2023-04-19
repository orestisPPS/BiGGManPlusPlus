//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FDSCHEMESPECS_H
#define UNTITLED_FDSCHEMESPECS_H

#include <map>
#include <tuple>
#include <vector>
#include <utility>
#include <stdexcept>
#include <iostream>
#include "../../PositioningInSpace/DirectionsPositions.h"
#include "FDScheme.h"
#include "../../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

using namespace std;
using namespace PositioningInSpace;

namespace LinearAlgebra {

    class FDSchemeSpecs {
    public:
        
        // Specifications of a first derivative finite difference scheme.
        // Use this constructor for first order equations.
        // Input: Map containing the type of the scheme and the order of the scheme at each direction
        //        For example : schemeTypeAndOrderAtDirectionFirstDerivative[One] = (Central, 2)
        //        SpaceEntityType: The space entity for which the scheme is being built (Axis, Plane, Volume)
        FDSchemeSpecs(map<Direction,tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirectionFirstDerivative,
                      SpaceEntityType &space);
        
        // Specifications of a first and second derivative finite difference scheme.
        // Use this constructor for equations up to second order.
        // Input: Maps containing the type of the scheme and the order of the scheme at each direction
        //        For example : schemeTypeAndOrderAtDirectionFirstDerivative[One]  = (Forward, 1)
        //                      schemeTypeAndOrderAtDirectionSecondDerivative[Two] = (Central, 2)
        //        SpaceEntityType: The space entity for which the scheme is being built (Axis, Plane, Volume)
        //        diagonalTermsCalculated: If true, the diagonal terms of the equation will be calculated with
        //                                 a central scheme of order Δx_i^2. If false or neglected the diagonal
        //                                 terms will not be added to the scheme
        FDSchemeSpecs(map<Direction,tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirectionFirstDerivative,
                      map<Direction,tuple<FDSchemeType, int>> schemeTypeAndOrderAtDirectionSecondDerivative,
                      SpaceEntityType &space, bool diagonalTermsCalculated = false);
        
        // Specifications of a first derivative finite difference scheme.
        // Use this constructor for first order equations where the scheme type and order are the same for all directions.
        FDSchemeSpecs(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder, SpaceEntityType &space);
        
        // Specifications of a first and second derivative finite difference scheme.
        // Use this constructor for equations up to second order where the scheme type and order are 
        // the same for all directions.
        FDSchemeSpecs(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
                      FDSchemeType secondDerivativeSchemeType, unsigned secondDerivativeOrder,
                      SpaceEntityType &space, bool diagonalTermsCalculated = false);
        
        ~FDSchemeSpecs();
        
        map<unsigned, map<Direction, tuple<FDSchemeType, int>>>* schemeTypeAndOrderAtDirectionForDerivativeOrder;
    private:
        void checkInput();
    };

} // LinearAlgebra

#endif //UNTITLED_FDSCHEMESPECS_H
