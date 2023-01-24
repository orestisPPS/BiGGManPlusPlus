//
// Created by hal9000 on 1/8/23.
//

#include "Line.h"

namespace PositioningInSpace {
    // One-dimensional space entity (line).
    Line::Line(PhysicalSpaceEntities lineType) : PhysicalSpaceEntity() {
        if (_checkInput(lineType))
            this->_type = lineType;
        else
            this->_type = NullSpace;
    }
    
    //Returns the axis where the line is located.
    const PhysicalSpaceEntities& Line::type() {
        return this->_type;
    }
    
    //Checks if the input is a valid Physical space entity. Should be : One_axis [x,r,ξ], Two_axis[y,θ,η], Three_axis.[z,φ,ζ]. 
    bool Line::_checkInput(PhysicalSpaceEntities type) {
        if (type == PhysicalSpaceEntities::One_axis   ||
            type == PhysicalSpaceEntities::Two_axis ||
            type == PhysicalSpaceEntities::Three_axis)
            return true;
        else
            return false;
    }
} // PositioningInSpace