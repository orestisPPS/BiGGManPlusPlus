//
// Created by hal9000 on 1/8/23.
//

#include "PhysicalSpaceEntity.h"

namespace PositioningInSpace {
    PhysicalSpaceEntity::PhysicalSpaceEntity( SpaceEntityType physicalSpace) {
        _type = physicalSpace;
    }
    
    //Copy constructor
    PhysicalSpaceEntity::PhysicalSpaceEntity(const PhysicalSpaceEntity& other) {
        _type = other._type;
    }
    
    //Move constructor
    PhysicalSpaceEntity::PhysicalSpaceEntity(PhysicalSpaceEntity&& other) noexcept {
        _type = other._type;
    }
    
    // = operator
    PhysicalSpaceEntity& PhysicalSpaceEntity::operator=(const PhysicalSpaceEntity& other) {
        _type = other._type;
        return *this;
    }
    
    const SpaceEntityType& PhysicalSpaceEntity::type() {
        return _type;
    }
    
    // PositioningInSpace
    
} // PositioningInSpace
