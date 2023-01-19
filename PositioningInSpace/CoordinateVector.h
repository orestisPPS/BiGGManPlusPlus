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
        CoordinateVector(vector<double> positionVector, PhysicalSpaceEntities physicalSpace);
        CoordinateVector(PhysicalSpaceEntities physicalSpace);
        
        //Initializes the CoordinateVector with the input vector of doubles in the input physical space
        void setCoordinateVector(const vector<double>& positionVector, PhysicalSpaceEntities physicalSpace);
        
        //Initializes a coordinate vector with NaN values in the input physical space
        void setCoordinateVector(PhysicalSpaceEntities &physicalSpace);
        
        vector<double> getCoordinateVectorInEntity(const PhysicalSpaceEntities &thisPhysicalSpace, PhysicalSpaceEntities physicalSpace);
        vector<double> getCoordinateVectorIn3D(const PhysicalSpaceEntities &thisPhysicalSpace);

        unsigned dimensions();
        
    private:
        vector<double> _positionVector;

    };

} // PositioningInSpace

#endif //UNTITLED_COORDINATEVECTOR_H
