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
    
    enum CoordinateType{
        Natural,
        Parametric,
        Templete
    };
    
    class CoordinateVector {
    public:
        CoordinateVector(vector<double> positionVector, PhysicalSpaceEntities physicalSpace);
        vector<double> getCoordinateVectorInEntity(const PhysicalSpaceEntities &thisPhysicalSpace, PhysicalSpaceEntities physicalSpace);
        vector<double> getCoordinateVectorIn3D(const PhysicalSpaceEntities &thisPhysicalSpace);
        unsigned dimensions();
        
    private:
        vector<double> _positionVector;
        void _setCoordinateVector(const vector<double>& positionVector, PhysicalSpaceEntities &physicalSpace);
    };

} // PositioningInSpace

#endif //UNTITLED_COORDINATEVECTOR_H
