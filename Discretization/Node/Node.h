//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_NODE_H
#define UNTITLED_NODE_H
#pragma once
#include "../Id/DiscreteEntityId.h"
#include <map>
#include <list>
#include "../../PositioningInSpace/CoordinateVector.h"
#include "../../DegreesOfFreedom/DegreeOfFreedom.h"
using namespace PositioningInSpace;

using namespace DegreesOfFreedom;
using namespace std;

namespace Discretization
{
    class Node {

    public:

        explicit Node();

        Discretization::DiscreteEntityId id;
                
        //Adds the input coordinate set type into the node coordinate vector map.
        //Initiated with input vector.
        void setPositionVector(vector<double> positionVector, CoordinateType type);


        //Adds a Natural coordinate set the node coordinate vector map.
        //Initiated with input vector.
        void setPositionVector(vector<double> positionVector);

        //Adds a coordinate set the node coordinate vector map.
        //The coordinates can be natural, parametric or template.
        //Initiated with NaN values.
        void setPositionVector(CoordinateType type);

        //Replaces the coordinate set of the input type with the input coordinate vector.
        //The coordinates can be natural, parametric or template.
        void changePositionVector(vector<double> positionVector, CoordinateType type);

        //Replaces the Natural Coordinate set of the input type with the input coordinate vector.
        //The coordinates can be natural, parametric or template.
        void changePositionVector(vector<double> positionVector);
        
        //Removes the input coordinate set from the node coordinate vector map.
        void removePositionVector(CoordinateType type);
        
        
        //Returns the natural position vector of the Node in a vector with 3 components
        vector<double> positionVector();

        //Returns the input position vector of the Node in a vector with 3 components
        vector<double> positionVector(CoordinateType type);
        
        //Returns a vector with the components of the natural position vector that exists
        //in the input physical space entity. 
        //The number of components is determined by the number of axes that define the input space. 
        //Use the other implementation to get the vector in the space that the problem is defined in.
        vector<double> positionVector(SpaceEntityType physicalSpace);

        //Returns a vector with the components of the input position vector that exists
        //in the input physical space entity. 
        //The number of components is determined by the number of axes that define the input space. 
        //Use the other implementation to get the vector in the space that the problem is defined in.
        vector<double> positionVector(CoordinateType type, SpaceEntityType physicalSpace);
        
        //Returns the number of components of the natural position vector
        unsigned positionVectorDimensions();
        
        //Returns the number of components of the input position vector
        unsigned positionVectorDimensions(CoordinateType type);
        
    private:
        map<CoordinateType, CoordinateVector> _position;
        
        list <DegreeOfFreedom*> *_degreesOfFreedom;
    };
}




#endif //UNTITLED_NODE_H


/*        

        //Returns the pointer of the degree of freedom of the given type and field
        DegreeOfFreedom *degreeOfFreedom(DOFType type, FieldType fieldType);
        // Adds a degree of freedom of the given type, field and value to the node
        void addDegreeOfFreedom(DOFType type, FieldType fieldType, double value);
        // Adds a degree of freedom of the given type and field to the node
        void addDegreeOfFreedom(DOFType type, FieldType fieldType);
        // Removes and deallocates the degree of freedom of the given type and field from the node
        void removeDegreeOfFreedom(DOFType type, FieldType fieldType);*/