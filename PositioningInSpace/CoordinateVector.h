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
        //Initializes the CoordinateVector with the input vector
        explicit CoordinateVector(vector<double> positionVector);
        
        //Initializes the CoordinateVector with the input physical space with NaN values
        explicit CoordinateVector(const SpaceEntityType &physicalSpace);
                
        //Initializes the CoordinateVector with the input vector
        void setCoordinateVector(vector<double> positionVector);
        
        //Returns a constant reference of the coordinate vector
        vector<double> *getCoordinateVector();
        
        //Returns the number of dimensions of the coordinate vector
        unsigned dimensions();
        
    private:
        vector<double>* _positionVector;

        //Initializes the coordinate vector according to the input physical space with NaN values
        static vector<double> initializeWithNaN(const SpaceEntityType &physicalSpace);
        
    };

} // PositioningInSpace

#endif //UNTITLED_COORDINATEVECTOR_H
