//
// Created by hal9000 on 1/8/23.
//

#include "CoordinateSystem.h"

#include <utility>

namespace PositioningInSpace {
    CoordinateSystem::CoordinateSystem(coordinateSystemType type) {
        _type = type;
    }

    coordinateSystemType CoordinateSystem::type() {
        return _type;
    }
    
    std::list<Direction> CoordinateSystem::axes() {
        return _axes;
    }
    
    void CoordinateSystem::setAxes(std::list<Direction> axes) {
            _axes = std::move(axes);
    }
        
    unsigned CoordinateSystem::dimensions() {
        return _axes.size();
    }
    
} // PositioningInSpace