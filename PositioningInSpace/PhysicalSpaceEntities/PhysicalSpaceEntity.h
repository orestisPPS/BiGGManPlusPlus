//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_PHYSICALSPACEENTITY_H
#define UNTITLED_PHYSICALSPACEENTITY_H

#include <list>
#include "../DirectionsPositions.h"
using namespace PositioningInSpace;
using namespace std;

namespace PositioningInSpace {

    enum SpaceEntityType {
        Axis,
        Plane,
        Volume,
        NullSpace
    };


    class PhysicalSpaceEntity {
     public:
        explicit PhysicalSpaceEntity(SpaceEntityType physicalSpace); 
        //Returns the type of the physical space
        const SpaceEntityType& type();
        
        //Returns the list of directions that define the space where the node exists (axis1, axis2, axis3, time)
        list<Direction> directions();
        
        //Copy constructor
        PhysicalSpaceEntity(const PhysicalSpaceEntity& other);
        //Move constructor
        PhysicalSpaceEntity(PhysicalSpaceEntity&& other) noexcept;
        // = operator
        PhysicalSpaceEntity& operator=(const PhysicalSpaceEntity& other);
     private:
         SpaceEntityType _type;
     };
} // PositioningInSpace

#endif //UNTITLED_PHYSICALSPACEENTITY_H
