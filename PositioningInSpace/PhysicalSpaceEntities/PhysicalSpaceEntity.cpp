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
    
    list<Direction> PhysicalSpaceEntity::directions() {
        list<Direction> directions;
        switch (_type) {
            case One_axis:
                directions.push_back(Direction::One);
                break;
            case Two_axis:
                directions.push_back(Direction::Two);
                break;
            case Three_axis:
                directions.push_back(Direction::Three);
                break;
            case OneTwo_plane:
                directions.push_back(Direction::One);
                directions.push_back(Direction::Two);
                break;
            case OneThree_plane:
                directions.push_back(Direction::One);
                directions.push_back(Direction::Three);
                break;
            case TwoThree_plane:
                directions.push_back(Direction::Two);
                directions.push_back(Direction::Three);
                break;
            case OneTwoThree_volume:    
                directions.push_back(Direction::One);
                directions.push_back(Direction::Two);
                directions.push_back(Direction::Three);
                break;
            case NullSpace:
                break;
        }
        return directions;
    }
    
} // PositioningInSpace
