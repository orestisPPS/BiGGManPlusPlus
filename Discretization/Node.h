//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_NODE_H
#define UNTITLED_NODE_H

#include "DiscreteEntityId.h"
#include <map>
#include <list>
#include "../PositioningInSpace/Coordinate.h"
using namespace PositioningInSpace;
#include "../DegreesOfFreedom/DegreeOfFreedom.h"
using namespace DegreesOfFreedom;
using namespace std;

namespace Dicretization
{
    class Node {

    public:

        Node();

        ~Node();

        DiscreteEntityId *id;

        //Returns the pointer to the coordinate with the given  type and direction
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
        void removeDegreeOfFreedom(DOFType type, FieldType fieldType);

    private:
        list <Coordinate*> *_coordinates;

        list <DegreeOfFreedom*> *_degreesOfFreedom;
    };
}




#endif //UNTITLED_NODE_H
