//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_COORDINATEVECTOR_H
#define UNTITLED_COORDINATEVECTOR_H

#include <vector>
#include <iostream>
#include <limits>
#include "PhysicalSpaceEntities/PhysicalSpaceEntity.h"

using namespace std;
using namespace PositioningInSpace;

namespace PositioningInSpace {
    
    enum CoordinateType{
        Natural,
        Parametric,
        Template
    };
    
    class CoordinateVector {
    public:
        CoordinateVector(vector<double> positionVector, const SpaceEntityType &physicalSpace);
        CoordinateVector(const SpaceEntityType &physicalSpace);
        
        //Initializes the CoordinateVector with the input vector of doubles in the input physical space
        static vector<double> setCoordinateVector(const vector<double>& positionVector, const SpaceEntityType &physicalSpace);
        //Initializes a coordinate vector with NaN values in the input physical space
        static vector<double> setCoordinateVector(const SpaceEntityType &physicalSpace);
        
        vector<double> getCoordinateVectorInEntity(const SpaceEntityType &thisPhysicalSpace, const SpaceEntityType &physicalSpace);
        vector<double> getCoordinateVectorIn3D(const SpaceEntityType &thisPhysicalSpace);

        unsigned dimensions();
        
    private:
        vector<double> _positionVector;

    };

} // PositioningInSpace

#endif //UNTITLED_COORDINATEVECTOR_H
