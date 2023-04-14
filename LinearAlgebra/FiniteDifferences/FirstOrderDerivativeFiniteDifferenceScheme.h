//
// Created by hal9000 on 4/13/23.
//

#ifndef UNTITLED_FIRSTORDERDERIVATIVEFINITEDIFFERENCESCHEME_H
#define UNTITLED_FIRSTORDERDERIVATIVEFINITEDIFFERENCESCHEME_H

#include "../../PositioningInSpace/DirectionsPositions.h"
#include "FDScheme.h"

using namespace PositioningInSpace;

namespace LinearAlgebra {
    class FirstOrderDerivativeFiniteDifferenceScheme {
    
    public:
        struct FirstOrderFDScheme;
        
        struct forward1;
        struct forward2;
        struct forward3;
        struct forward4;
        struct forward5;
        
        struct backward1;
        struct backward2;
        struct backward3;
        struct backward4;
        struct backward5;
        
        struct central2;
        struct central4;
        struct central6;
        
        struct FiniteDifferenceSchemeTypeToRelativePositions;
        
        static map<unsigned, map<FiniteDifferenceSchemeType, vector<Position>>> positionsForFDSchemeTypeAndDimensions();
    };

} // LinearAlgebra

#endif //UNTITLED_FIRSTORDERDERIVATIVEFINITEDIFFERENCESCHEME_H
