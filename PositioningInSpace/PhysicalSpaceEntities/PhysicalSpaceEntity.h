//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_PHYSICALSPACEENTITY_H
#define UNTITLED_PHYSICALSPACEENTITY_H

#include "../DirectionsPositions.h"
#include "../CoordinateVector.h"

using namespace PositioningInSpace;

namespace PositioningInSpace {

     class PhysicalSpaceEntity {
     public:
         explicit PhysicalSpaceEntity() : _type(NullSpace) {};
         virtual const PhysicalSpaceEntities  &type();
     private:
         PhysicalSpaceEntities _type;
     };
} // PositioningInSpace

#endif //UNTITLED_PHYSICALSPACEENTITY_H
