//
// Created by hal9000 on 1/8/23.
//

#include "Plane.h"

namespace PositioningInSpace {
        
        // Two-dimensional space entity (plane).
        Plane::Plane(PhysicalSpaceEntities type) : PhysicalSpaceEntity() {
            if (_checkInput(type))
                _type = type;
            else
                _type = NullSpace;
        }
        //Returns the plane. OneTwoPlane [xy, ξη, rθ], OneThreePlane [xz etc. ], TwoThreePlane [yz etc.].
        const PhysicalSpaceEntities& Plane::type() {
            return _type;
        }
        
        //Checks if the input is a valid Physical space entity. Should be : OneTwoPlane [xy, ξη, rθ], OneThreePlane [xz etc. ], TwoThreePlane [yz etc.].
        bool Plane::_checkInput(PhysicalSpaceEntities type) {
            if (type == PhysicalSpaceEntities::OneTwo_plane   ||
                type == PhysicalSpaceEntities::OneThree_plane ||
                type == PhysicalSpaceEntities::TwoThree_plane)
                return true;
            else
                return false;
        }
} // PositioningInSpace