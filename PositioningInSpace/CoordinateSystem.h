//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_COORDINATESYSTEM_H
#define UNTITLED_COORDINATESYSTEM_H

#include <list>
#include <iostream>

#include "DirectionsPositions.h"
#include "PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "PhysicalSpaceEntities/Line.h"
#include "PhysicalSpaceEntities/Plane.h"
#include "PhysicalSpaceEntities/Volume.h"

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
        
        //The physical space entity that is defined by this coordinate system.
        PhysicalSpaceEntity space;
        
        //The type of the coordinate system. Can be Cartesian, polar etc.)
        coordinateSystemType type();
        
        //The axes of the coordinate system.
        std::list<Direction> axes();
        
        void setAxes(std::list<Direction> axes);
        
        //The number of dimensions of the coordinate system.
        unsigned dimensions();
    private:
        std::list<Direction> _axes;
        coordinateSystemType _type;
        PhysicalSpaceEntity _findPhysicalSpace();
    };

};// PositioningInSpace

#endif //UNTITLED_COORDINATESYSTEM_H
