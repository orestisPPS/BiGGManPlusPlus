//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_PHYSICALSPACEENTITY_H
#define UNTITLED_PHYSICALSPACEENTITY_H

#include "../DirectionsPositions.h"
using namespace PositioningInSpace;

namespace PositioningInSpace {

    enum SpaceEntityType {
        //Axis 1. Can be x,ξ,θ
        One_axis,
        //Axis 2, Can be y,η,φ
        Two_axis,
        //Axis_3, Can be z,ζ,r
        Three_axis,
        //The physical space defined by Axis 1 and Axis 2.
        OneTwo_plane,
        //The physical space defined by Axis 1 and Axis 3.
        OneThree_plane,
        //The physical space defined by Axis 2 and Axis 3.
        TwoThree_plane,
        //The 3D physical space.
        OneTwoThree_volume,
        //0D physical space. Enter the void.
        NullSpace
    };


    class PhysicalSpaceEntity {
     public:
        explicit PhysicalSpaceEntity(SpaceEntityType physicalSpace); 
        const SpaceEntityType& type();
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
