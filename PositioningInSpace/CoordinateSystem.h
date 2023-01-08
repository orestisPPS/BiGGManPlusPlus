//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_COORDINATESYSTEM_H
#define UNTITLED_COORDINATESYSTEM_H

#include <list>

#include "DirectionsPositions.h"
using namespace PositioningInSpace;

namespace PositioningInSpace {

    enum coordinateSystemType {
        cartesian,
        spherical,
        cylindrical,
        polar
    };
       
    class CoordinateSystem {
    public:
        CoordinateSystem(coordinateSystemType type);
        coordinateSystemType type();
        std::list<Direction> axes();
        void setAxes(std::list<Direction> axes);
        unsigned dimensions();
    private:
        std::list<Direction> _axes;
        coordinateSystemType _type;
    };

};// PositioningInSpace

#endif //UNTITLED_COORDINATESYSTEM_H
