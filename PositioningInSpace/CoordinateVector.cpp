//
// Created by hal9000 on 1/8/23.
//

#include "CoordinateVector.h"

#include <utility>

namespace PositioningInSpace {
    
    CoordinateVector::CoordinateVector(vector<double> positionVector){
        _positionVector = new vector<double>(std::move(positionVector));
    }
    
    CoordinateVector::CoordinateVector(const SpaceEntityType &physicalSpace){
        _positionVector = new vector<double>(std::move(initializeWithNaN(physicalSpace)));
    }
    
    //Initializes the coordinate vector according to the input physical space with NaN values
    vector<double> CoordinateVector::initializeWithNaN(const SpaceEntityType &physicalSpace) {
        vector<double> positionVector;
        switch (physicalSpace) {
            case Axis:
                positionVector = {numeric_limits<double>::quiet_NaN()};
                break;
            case Plane:
                positionVector = {numeric_limits<double>::quiet_NaN(), numeric_limits<double>::quiet_NaN()};
                break;
            case Volume:
                positionVector = {numeric_limits<double>::quiet_NaN(), numeric_limits<double>::quiet_NaN(),
                                  numeric_limits<double>::quiet_NaN()};
                break;
            case NullSpace:
                positionVector = {};
                break;
        }
        return positionVector;
    }
    
    //Initializes the CoordinateVector with the input vector
    void CoordinateVector::setCoordinateVector(vector<double> positionVector) {
        if ( dimensions() != positionVector.size() ) {
            throw invalid_argument("The input vector must have the same number of dimensions as the CoordinateVector");
        }
        for (int i = 0; i < dimensions(); ++i) {
            (*_positionVector)[i] = positionVector[i];
        }
    }
    
    //Returns the coordinate vector
    vector<double> *CoordinateVector::getCoordinateVector() {
        return _positionVector;
    }
    
    unsigned CoordinateVector::dimensions() {
        return _positionVector->size();
    }
    
} // PositioningInSpace
