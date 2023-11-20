//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FINITEDIFFERENCESCHEMEORDER_H
#define UNTITLED_FINITEDIFFERENCESCHEMEORDER_H

#include <map>
#include <tuple>
#include <vector>
#include <utility>
#include <stdexcept>
#include <iostream>
#include "../../PositioningInSpace/DirectionsPositions.h"
#include "../../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "FDSchemeType.h"

using namespace std;
using namespace PositioningInSpace;


/*//Finite Difference Scheme Type
//The categorization is based on the location of the points used for approximation with respect to the point at
// which the derivative is being calculated.
enum FDSchemeType{
    Forward,
    Backward,
    Central,
    Mixed
};*/

namespace LinearAlgebra {

    
    class FiniteDifferenceSchemeOrder {
    public:
        
        // Specifications of a first derivative finite difference scheme.
        // Use this constructor for first order equations.
        // Input: Map containing the type of the scheme and the order of the scheme at each direction
        //        For example : schemeTypeAndOrderAtDirectionFirstDerivative[One] = (Central, 2)
        //        SpaceEntityType: The space entity for which the scheme is being built (Axis, Plane, Volume)
        explicit FiniteDifferenceSchemeOrder(const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirectionFirstDerivative);
        
        // Specifications of a first and second derivative finite difference scheme.
        // Use this constructor for equations up to second order.
        // Input: Maps containing the type of the scheme and the order of the scheme at each direction
        //        For example : schemeTypeAndOrderAtDirectionFirstDerivative[One]  = (Forward, 1)
        //                      schemeTypeAndOrderAtDirectionSecondDerivative[Two] = (Central, 2)
        //        SpaceEntityType: The space entity for which the scheme is being built (Axis, Plane, Volume)
        //        diagonalTermsCalculated: If true, the diagonal terms of the equation will be calculated with
        //                                 a central scheme of order Δx_i^2. If false or neglected the diagonal
        //                                 terms will not be added to the scheme
        FiniteDifferenceSchemeOrder(const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirectionFirstDerivative,
                                    const map<Direction,tuple<FDSchemeType, int>>& schemeTypeAndOrderAtDirectionSecondDerivative,
                                    bool diagonalTermsCalculated = false);
        
        // Specifications of a first derivative finite difference scheme.
        // Use this constructor for first order equations where the scheme type and order are the same for all directions.
        FiniteDifferenceSchemeOrder(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder, const vector<Direction> &directions);
        
        
        FiniteDifferenceSchemeOrder(unsigned short firstDerivativeOrder, const vector<Direction> &directions);
        
        
        
        // Specifications of a first and second derivative finite difference scheme.
        // Use this constructor for equations up to second order where the scheme type and order are 
        // the same for all directions.
        FiniteDifferenceSchemeOrder(FDSchemeType firstDerivativeSchemeType, unsigned firstDerivativeOrder,
                                    FDSchemeType secondDerivativeSchemeType, unsigned secondDerivativeOrder,
                                    const vector<Direction> &directions, bool diagonalTermsCalculated = false);

        // Specifications of a first and second derivative finite difference scheme.
        // Use this constructor for equations up to second order where the scheme type and order are 
        // the same for all directions.
        FiniteDifferenceSchemeOrder(unsigned firstDerivativeOrder, unsigned secondDerivativeOrder,
                                    const vector<Direction> &directions, bool diagonalTermsCalculated = false);
        
        ~FiniteDifferenceSchemeOrder();
        
        map<unsigned, map<Direction, tuple<FDSchemeType, int>>>* schemeTypeAndOrderAtDirectionForDerivativeOrder;
        
        unsigned getErrorOrderOfSchemeTypeForDerivative(unsigned derivativeOrder) const;
    
        bool schemeTypeFixed;

        unsigned int getErrorForDerivativeOfArbitraryScheme(unsigned int derivativeOrder) const;

    private:
        void checkInput() const;

    };

} // LinearAlgebra

#endif //UNTITLED_FINITEDIFFERENCESCHEMEORDER_H
