//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_COORDINATEVECTOR_H
#define UNTITLED_COORDINATEVECTOR_H

#include <vector>
#include <iostream>
#include "PhysicalSpaceEntities/PhysicalSpaceEntity.h"
using namespace std;
using namespace PositioningInSpace;

namespace PositioningInSpace {

    class CoordinateVector {
    public:
        vector<double> getCoordinateVectorInEntity(const PhysicalSpaceEntities &thisPhysicalSpace, PhysicalSpaceEntities physicalSpace);
        vector<double> getCoordinateVectorIn3D(const PhysicalSpaceEntities &thisPhysicalSpace);
        void setCoordinateVector(vector<double> coordinateVector, PhysicalSpaceEntities &physicalSpace);
        unsigned dimensions();
    private:
        vector<double> _positionVector;
    };

} // PositioningInSpace

#endif //UNTITLED_COORDINATEVECTOR_H
