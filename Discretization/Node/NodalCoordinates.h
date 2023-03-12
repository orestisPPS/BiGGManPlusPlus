//
// Created by hal9000 on 2/5/23.
//

#ifndef UNTITLED_NODALCOORDINATES_H
#define UNTITLED_NODALCOORDINATES_H

#include <map>
#include "../../PositioningInSpace/CoordinateVector.h"
#include "../../PositioningInSpace/DirectionsPositions.h"
namespace Discretization {

    class NodalCoordinates {
    public:
        NodalCoordinates();
        
        const double& operator()(unsigned i) const;
        
        const double& operator()(CoordinateType type, unsigned i) const;
        
        //Adds the input coordinate set type into the node coordinate vector map.
        //Initiated with input vector.
        void addPositionVector(vector<double> positionVector, CoordinateType type);

        //Adds a Natural coordinate set the node coordinate vector map.
        //Initiated with input vector.
        void addPositionVector(vector<double> positionVector);
        
        //Adds a coordinate set the node coordinate vector map.
        void addPositionVector(CoordinateType type);
        
        //Replaces the coordinate set of the input type with the input coordinate vector.
        //The coordinates can be natural, parametric or template.
        void setPositionVector(vector<double> positionVector, CoordinateType type);

        //Replaces the Natural Coordinate set of the input type with the input coordinate vector.
        //The coordinates can be natural, parametric or template.
        void setPositionVector(vector<double> positionVector);

        //Removes the input coordinate set from the node coordinate vector map.
        void removePositionVector(CoordinateType type);        
        
        //Returns the natural position vector of the Node
        const vector<double>& positionVector();
        
        //Returns a pointer to the natural position vector of the Node
        vector<double>* positionVectorPtr();

        //Returns the input position vector of the Node
        const vector<double>& positionVector(CoordinateType type);
        
        //Returns a pointer to the input position vector of the Node
        vector<double>* positionVectorPtr(CoordinateType type);
        
        //Returns the number of components of the natural position vector
        unsigned size();

        //Returns the number of components of the input position vector
        unsigned size(CoordinateType type);
        
    private:
        map<CoordinateType, CoordinateVector> _positionVectors;
    };

} // Discretization

#endif //UNTITLED_NODALCOORDINATES_H
