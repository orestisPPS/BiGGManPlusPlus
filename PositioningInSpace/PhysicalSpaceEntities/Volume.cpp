//
// Created by hal9000 on 1/8/23.
//

#include "Volume.h"

namespace PositioningInSpace {
    // Three-dimensional space entity (plane).
    Volume::Volume(PhysicalSpaceEntities volumeType) : PhysicalSpaceEntity() {
        if (_checkInput(volumeType))
            this->_type = volumeType;
        else
            this->_type = NullSpace;
    }

    const PhysicalSpaceEntities& Volume::type() {
        return this->_type;
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