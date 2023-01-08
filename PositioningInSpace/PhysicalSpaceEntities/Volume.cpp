//
// Created by hal9000 on 1/8/23.
//

#include "Volume.h"

namespace PositioningInSpace {
        // Three-dimensional space entity (plane).
        Volume::Volume(PhysicalSpaceEntities type) : PhysicalSpaceEntity() {
            if (_checkInput(type))
                _type = type;
            else
                _type = NullSpace;
        }
    
        PhysicalSpaceEntities Volume::type() {
            return _type;
        }
    
        bool Volume::_checkInput(PhysicalSpaceEntities type) {
            switch (type) {
                case PhysicalSpaceEntities::OneTwoThree_volume:
                    return true;
                default:
                    return false;
            }
        }
    
    } // PositioningInSpace
} // PositioningInSpace