//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_PHYSICALSPACEENTITY_H
#define UNTITLED_PHYSICALSPACEENTITY_H

#include "../DirectionsPositions.h"
using namespace PositioningInSpace;

namespace PositioningInSpace {

     class PhysicalSpaceEntity {
     public:
         PhysicalSpaceEntity();
         virtual PhysicalSpaceEntities type();
    };

} // PositioningInSpace

#endif //UNTITLED_PHYSICALSPACEENTITY_H
