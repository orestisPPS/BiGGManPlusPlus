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
#include "../../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"
#include "../../DegreesOfFreedom/DegreeOfFreedom.h"
using namespace PositioningInSpace;

using namespace DegreesOfFreedom;
using namespace std;

namespace Discretization
{
    class Node {

    public:

        Node(PhysicalSpaceEntity space);

        Discretization::DiscreteEntityId id;
        
        PhysicalSpaceEntity space;
                
        //Returns the position vector of the Node in a vector with 3 components
        vector<double> positionVector();
        
        //Returns a vector with the components of the position vector that exist
        //in the input physical space entity. 
        //The number of components is determined by the number of axes that define the input space. 
        //If there is  no space entity preference use the other implementation.
        vector<double> positionVector(PhysicalSpaceEntities physicalSpace);
        
        
    private:
        CoordinateVector _position;
        
        list <DegreeOfFreedom*> *_degreesOfFreedom;
    };
}




#endif //UNTITLED_NODE_H


/*        //Returns the pointer to the coordinate with the given  type and direction
        Coordinate *coordinate(CoordinateType type, Direction direction);
        // Adds a coordinate of the given type, direction and value to the node
        void addCoordinate(CoordinateType type, Direction direction, double value);
        // Adds a coordinate of the given type and direction to the node
        void addCoordinate(CoordinateType type, Direction direction);
        // Removes and deallocates the coordinate of the given type and direction from the node
        void removeCoordinate(CoordinateType type, Direction direction);

        //Returns the pointer of the degree of freedom of the given type and field
        DegreeOfFreedom *degreeOfFreedom(DOFType type, FieldType fieldType);
        // Adds a degree of freedom of the given type, field and value to the node
        void addDegreeOfFreedom(DOFType type, FieldType fieldType, double value);
        // Adds a degree of freedom of the given type and field to the node
        void addDegreeOfFreedom(DOFType type, FieldType fieldType);
        // Removes and deallocates the degree of freedom of the given type and field from the node
        void removeDegreeOfFreedom(DOFType type, FieldType fieldType);*/