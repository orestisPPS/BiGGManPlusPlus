//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_NODE_H
#define UNTITLED_NODE_H

#include "../Id/DiscreteEntityId.h"
#include "NodalCoordinates.h"
//#include "../../DegreesOfFreedom/DegreeOfFreedom.h"

using namespace PositioningInSpace;
using namespace std;

namespace Discretization
{
    class Node {

    public:

        Node();

        Discretization::DiscreteEntityId id;
                
        NodalCoordinates coordinates;

        //vector<DegreeOfFreedom*> *_degreesOfFreedom;
        
    private:
               

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