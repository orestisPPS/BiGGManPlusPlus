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
        FDSchemeSpecs(map<Direction,tuple<FiniteDifferenceSchemeType, int>> schemeTypeAndOrderAtDirection,
                      PhysicalSpaceEntity &space);
        
        map<Direction, tuple<FiniteDifferenceSchemeType, int>> schemeTypeAndOrderAtDirection;
        PhysicalSpaceEntity &space;
    private:
        bool checkInput();
    };

} // LinearAlgebra

#endif //UNTITLED_FDSCHEMESPECS_H
